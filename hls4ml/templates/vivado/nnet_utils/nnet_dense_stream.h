#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"
#include "hls_stream.h"
#include <math.h>
#include <assert.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_wrapper(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]
) {
    #pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense(
    hls::stream<data_T> &data_stream,
    hls::stream<res_T>  &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out])
{
    typename data_T::value_type data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=res complete

    DataPrepare: for(int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_in / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
        DataPack: for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    dense_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

    ResWrite: for(unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        if (CONFIG_T::n_out / res_T::size > 1) {
            #pragma HLS PIPELINE
        }
        res_T res_pack;
        #pragma HLS DATA_PACK variable=res_pack
        ResPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_ss(
      hls::stream<data_T> data[1],
      hls::stream<res_T>  res[1],
      typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
      typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {
      
      // Check the validation of the reuse factor
      const int reuse = (CONFIG_T::reuse_factor >= CONFIG_T::n_out) ? CONFIG_T::n_out : CONFIG_T::reuse_factor;
      if(CONFIG_T::reuse_factor > CONFIG_T::n_out){
          std::cout <<"Change the reuse factor in dense_ss to max value "<<CONFIG_T::n_out<<std::endl;
      }
      else {
          assert((CONFIG_T::n_out % reuse == 0) && "The current reuse factor is not allowed");
      }
      
      const int block_factor = DIV_ROUNDUP(CONFIG_T::n_out,reuse);
      #pragma HLS ARRAY_RESHAPE variable=weights block factor=CONFIG_T::n_out
      #pragma HLS ARRAY_PARTITION variable=biases complete
      
      typename CONFIG_T::accum_t acc[block_factor][reuse];
      #pragma HLS ARRAY_PARTITION variable=acc complete dim=0
      
      InitAccum:
      for (int iacc = 0; iacc < reuse; iacc++) {
          #pragma HLS UNROLL
          for (int iacc2 = 0; iacc2 < block_factor; iacc2++) {
            #pragma HLS UNROLL
            acc[iacc2][iacc] = (typename CONFIG_T::accum_t) biases[iacc*block_factor+iacc2];
          }
      }
    
     for(int i_in = 0; i_in < CONFIG_T::n_in; i_in++) {
        #pragma HLS PIPELINE II=reuse
        data_T tmpdata = data[0].read();
        for (int iacc = 0; iacc < reuse; iacc++) {
          #pragma HLS UNROLL
          for (int iacc2 = 0; iacc2 < block_factor; iacc2++) {
            #pragma HLS UNROLL
            unsigned w_index  =  i_in + (CONFIG_T::n_in*(iacc*block_factor+iacc2)); 
            acc[iacc2][iacc] += product_dense<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(tmpdata, weights[w_index]);
          }
      }
     }
     ResWrite:for (int iacc = 0; iacc < reuse; iacc++) {
          #pragma HLS UNROLL
          for (int iacc2 = 0; iacc2 < block_factor; iacc2++) {
            #pragma HLS UNROLL
            res_T tmpres = (res_T)acc[iacc2][iacc];
            res[0].write(tmpres);
          }
     }
}


}

#endif
