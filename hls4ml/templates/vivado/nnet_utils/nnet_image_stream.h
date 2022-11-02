#ifndef NNET_IMAGE_STREAM_H_
#define NNET_IMAGE_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void resize_nearest(
    hls::stream<data_T> &image,
    hls::stream<data_T> &resized
) {
	assert(CONFIG_T::new_height % CONFIG_T::height == 0);
	assert(CONFIG_T::new_width % CONFIG_T::width == 0);
	constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
	constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;
	
	
	ImageHeight: for (unsigned h = 0; h < CONFIG_T::height; h++) {
		#pragma HLS PIPELINE
	
		data_T data_in_row[CONFIG_T::width];
		
		// Read a row
		ImageWidth: for (unsigned i = 0; i < CONFIG_T::width; i++) {
			#pragma HLS UNROLL
			
			data_T  in_data = image.read();
			
			ImageChan: for (unsigned j = 0; j < CONFIG_T::n_chan; j++) {
				#pragma HLS UNROLL
				
				data_in_row[i][j] = in_data[j];
			}
		}
		
		ResizeHeight: for (unsigned i = 0; i <ratio_height; i++) {
			#pragma HLS UNROLL
			
			ImageWidth2: for (unsigned l = 0; l < CONFIG_T::width; l++) {
				#pragma HLS UNROLL
				
				ResizeWidth: for (unsigned j = 0; j < ratio_width; j++) {
					#pragma HLS UNROLL
				
					data_T out_data;
					#pragma HLS DATA_PACK variable=out_data
				
					ResizeChan: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
						#pragma HLS UNROLL
					
						out_data[k] = data_in_row[l][k];
					}
					
					resized.write(out_data);   
				}
			}
		}
	}
}

// --------------------------------------Single Stream--------------------------------------------

template<class data_T, typename CONFIG_T>
void resize_nearest_ss(
    hls::stream<data_T> &image,
    hls::stream<data_T> &resized
) {
	assert(CONFIG_T::new_height % CONFIG_T::height == 0);
	assert(CONFIG_T::new_width % CONFIG_T::width == 0);
	constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
	constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;
	
	data_T data_in_row[CONFIG_T::width][CONFIG_T::n_chan];
	
	// Read a row
	ImageHeight: for (unsigned h = 0; h < CONFIG_T::height; h++) {
		#pragma HLS PIPELINE
		
		ImageWidth: for (unsigned i = 0; i < CONFIG_T::width; i++) {
			#pragma HLS UNROLL
			
			ImageChan: for (unsigned j = 0; j < CONFIG_T::n_chan; j++) {
				#pragma HLS UNROLL
				data_T in_data = image.read();
				data_in_row[i][j] = in_data;
			}
		}
		
		// Write with a ratio in height and width
		ResizeHeight: for (unsigned i = 0; i <ratio_height; i++) {
			#pragma HLS UNROLL
			
			ImageWidth2: for (unsigned l = 0; l < CONFIG_T::width; l++) {
				#pragma HLS UNROLL
				
				ResizeWidth: for (unsigned j = 0; j < ratio_width; j++) {
					#pragma HLS UNROLL
				
					ResizeChan: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
						#pragma HLS UNROLL
						data_T out_data = data_in_row[l][k];
						resized.write(out_data);   
					}
					
					
				}
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//for switch
template<class data_T, typename CONFIG_T>
void resize_nearest_single(
    hls::stream<data_T> image[1],
    hls::stream<data_T> resized[1]
) {
	assert(CONFIG_T::new_height % CONFIG_T::height == 0);
	assert(CONFIG_T::new_width % CONFIG_T::width == 0);
	constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
	constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;
	constexpr unsigned ii = ratio_height * ratio_width;
	
	data_T data_in_row[CONFIG_T::width][CONFIG_T::n_chan];
	
	ImageHeight: for (unsigned h = 0; h < CONFIG_T::height; h++) {
		#pragma HLS PIPELINE II=CONFIG_T::new_width*CONFIG_T::n_chan
		ImageWidth: for (unsigned i = 0; i < CONFIG_T::width; i++) {
			ReadData: for(unsigned j = 0; j < CONFIG_T::n_chan ; j++){
			
			#pragma HLS loop_flatten
				data_in_row[i][j] = image[0].read();
			}
		}
		
		ResizeHeight: for (unsigned i = 0; i <ratio_height; i++) {
			ImageWidth2: for (unsigned l = 0; l < CONFIG_T::width; l++) {
				ResizeWidth: for (unsigned j = 0; j < ratio_width; j++) {
					ResizeChan: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
						#pragma HLS loop_flatten
						data_T out_data = data_in_row[l][k];
						resized[0].write(out_data);   
					}
				}
			}
		}
	}
}

template<class data_T, typename CONFIG_T>
void resize_nearest_array(
    hls::stream<data_T> image[CONFIG_T::n_chan],
    hls::stream<data_T> resized[CONFIG_T::n_chan]
) {
	assert(CONFIG_T::new_height % CONFIG_T::height == 0);
	assert(CONFIG_T::new_width % CONFIG_T::width == 0);
	constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
	constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;
	constexpr unsigned ii = ratio_height * ratio_width;
	
	data_T data_in_row[CONFIG_T::width][CONFIG_T::n_chan];
	
	ImageHeight: for (unsigned h = 0; h < CONFIG_T::height; h++) {
		#pragma HLS PIPELINE II=CONFIG_T::new_width
		ImageWidth: for (unsigned i = 0; i < CONFIG_T::width; i++) {
			ReadData: for(unsigned j = 0; j < CONFIG_T::n_chan ; j++){
			
			#pragma HLS loop_flatten
				data_in_row[i][j] = image[j].read();
			}
		}
		
		ResizeHeight: for (unsigned i = 0; i <ratio_height; i++) {
			ImageWidth2: for (unsigned l = 0; l < CONFIG_T::width; l++) {
				ResizeWidth: for (unsigned j = 0; j < ratio_width; j++) {
					ResizeChan: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
						#pragma HLS loop_flatten
						data_T out_data = data_in_row[l][k];
						resized[k].write(out_data);   
					}
				}
			}
		}
	}
}


template <class data_T, typename CONFIG_T>
void resize_nearest_switch(
    hls::stream<data_T> image[CONFIG_T::data_transfer_out],
    hls::stream<data_T>  resized[CONFIG_T::data_transfer_out]
) {
    #pragma HLS inline region
    if(CONFIG_T::data_transfer_out == 1){
        resize_nearest_single<data_T, CONFIG_T>(image, resized);
    }else {
        resize_nearest_array<data_T, CONFIG_T>(image, resized);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


}

#endif
