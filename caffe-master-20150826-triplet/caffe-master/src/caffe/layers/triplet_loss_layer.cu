#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[1]->gpu_data(),  // Anchored
      bottom[0]->gpu_data(),  // Positive
      diff_AP.mutable_gpu_data());  // A-P
  caffe_gpu_powx(
      count,
      diff_AP.mutable_gpu_data(),  // A_i-P_i
      Dtype(2),
      diff_sq_AP.mutable_gpu_data());  // (A_i-P_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_AP.gpu_data(),  // (A_i-P_i)^2
      summer_vec_AP.gpu_data(),
      Dtype(0.0),
      dist_sq_AP.mutable_gpu_data());  // \Sum (A_i-P_i)^2


  caffe_gpu_sub(
      count,
      bottom[1]->gpu_data(),  // Anchored
      bottom[2]->gpu_data(),  // Negative
      diff_AN.mutable_gpu_data());  // A-N
  caffe_gpu_powx(
      count,
      diff_AN.mutable_gpu_data(),  // A_i-N_i
      Dtype(2),
      diff_sq_AN.mutable_gpu_data());  // (A_i-N_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_AN.gpu_data(),  // (A_i-N_i)^2
      summer_vec_AN.gpu_data(),
      Dtype(0.0),
      dist_sq_AN.mutable_gpu_data());  // \Sum (A_i-N_i)^2


  caffe_gpu_sub(
      count,
      bottom[2]->gpu_data(),  // Negative
      bottom[0]->gpu_data(),  // positive
      diff_NP.mutable_gpu_data());  // N-P
  caffe_gpu_powx(
      count,
      diff_NP.mutable_gpu_data(),  // N_i-P_i
      Dtype(2),
      diff_sq_NP.mutable_gpu_data());  // (N_i-P_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_NP.gpu_data(),  // (N_i-P_i)^2
      summer_vec_NP.gpu_data(),
      Dtype(0.0),
      dist_sq_NP.mutable_gpu_data());  // \Sum (N_i-P_i)^2










  Dtype alpha_param = this->layer_param_.triplet_loss_param().margin();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
	loss += std::max(dist_sq_AP.cpu_data()[i]-dist_sq_AN.cpu_data()[i]+alpha_param, Dtype(0.0));
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void CLLForward_0(const int count, const int channels,
    const Dtype alpha_param, const Dtype alpha, const Dtype* diff_AP, const Dtype* dist_sq_AP,const Dtype* dist_sq_AN,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (dist_sq_AP[n]-dist_sq_AN[n]+alpha_param>0) {  
      bottom_diff[i] = alpha * diff_AP[i];
    } else {  // dissimilar pairs
        bottom_diff[i] = 0;
      }
  }
}

template <typename Dtype>
__global__ void CLLForward_1(const int count, const int channels,
    const Dtype alpha_param, const Dtype alpha, const Dtype* diff_NP, const Dtype* dist_sq_AP,const Dtype* dist_sq_AN,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (dist_sq_AP[n]-dist_sq_AN[n]+alpha_param>0) {  
      bottom_diff[i] = alpha * diff_NP[i];
    } else {  // dissimilar pairs
        bottom_diff[i] = 0;
      }
  }
}



template <typename Dtype>
__global__ void CLLForward_2(const int count, const int channels,
    const Dtype alpha_param, const Dtype alpha, const Dtype* diff_AN, const Dtype* dist_sq_AP,const Dtype* dist_sq_AN,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (dist_sq_AP[n]-dist_sq_AN[n]+alpha_param>0) {  
      bottom_diff[i] = alpha * diff_AN[i];
    } else {  // dissimilar pairs
        bottom_diff[i] = 0;
      }
  }
}








template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 3; ++i) 
  {
    if (propagate_down[i]) 
	{
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype alpha_param = this->layer_param_.triplet_loss_param().margin();
      const Dtype sign = (i == 0) ? -1 : 1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
	  if(i==0)
	  {
		 //   CUDA_KERNEL_LOOP(j, count) 
			//{
			//	int n = j / channels;  // the num index, to access y and dist_sq
			//	if (dist_sq_AP.gpu_data()[n]-dist_sq_AN.gpu_data()[n]+alpha_param>0) 
			//	{  
			//		bottom[i]->mutable_gpu_diff()[n] = alpha * diff_AP.gpu_data()[n];
			//	} 
			//	else 
			//	{  // dissimilar pairs
			//		bottom[i]->mutable_gpu_diff()[n] = 0;
			//	}
			//}
		  CLLForward_0<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, alpha_param, alpha,
          diff_AP.gpu_data(),  // pair similarity 0 or 1
          dist_sq_AP.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_AN.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
		  CUDA_POST_KERNEL_CHECK;



		  //CLLForward_0<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    //      count, channels, alpha_param, alpha,
    //      diff_AP.gpu_data,  // pair similarity 0 or 1
    //      dist_sq_AP.gpu_data(),  // the cached eltwise difference between a and b
    //      dist_sq_AN.gpu_data(),  // the cached square distance between a and b
    //      bottom[i]->mutable_gpu_diff());
	  }
	  else if(i == 1)
	  {
		  //CLLForward_1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    //      count, channels, alpha_param, alpha,
    //      diff_NP.gpu_data,  // pair similarity 0 or 1
    //      dist_sq_AP.gpu_data(),  // the cached eltwise difference between a and b
    //      dist_sq_AN.gpu_data(),  // the cached square distance between a and b
    //      bottom[i]->mutable_gpu_diff());
		 // 	CUDA_KERNEL_LOOP(j, count) 
			//{
			//	int n = j / channels;  // the num index, to access y and dist_sq
			//	if (dist_sq_AP.gpu_data()[n]-dist_sq_AN.gpu_data()[n]+alpha_param>0) 
			//	{  
			//		bottom[i]->mutable_gpu_diff()[n] = alpha * diff_NP.gpu_data()[n];
			//	} 
			//	else 
			//	{  // dissimilar pairs
			//		bottom[i]->mutable_gpu_diff()[n] = 0;
			//	}
			//}
		  CLLForward_1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, alpha_param, alpha,
          diff_NP.gpu_data(),  // pair similarity 0 or 1
          dist_sq_AP.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_AN.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
		  CUDA_POST_KERNEL_CHECK;

	  }
	  else
	  {
		  //CLLForward_2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    //      count, channels, alpha_param, alpha,
    //      diff_AN.gpu_data,  // pair similarity 0 or 1
    //      dist_sq_AP.gpu_data(),  // the cached eltwise difference between a and b
    //      dist_sq_AN.gpu_data(),  // the cached square distance between a and b
    //      bottom[i]->mutable_gpu_diff());
		 // 	CUDA_KERNEL_LOOP(j, count) 
			//{
			//	int n = j / channels;  // the num index, to access y and dist_sq
			//	if (dist_sq_AP.gpu_data()[n]-dist_sq_AN.gpu_data()[n]+alpha_param>0) 
			//	{  
			//		bottom[i]->mutable_gpu_diff()[n] = alpha * diff_AN.gpu_data()[n];
			//	} 
			//	else 
			//	{  // dissimilar pairs
			//		bottom[i]->mutable_gpu_diff()[n] = 0;
			//	}
			//}
		  CLLForward_2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, alpha_param, alpha,
          diff_AN.gpu_data(),  // pair similarity 0 or 1
          dist_sq_AP.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_AN.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
		  CUDA_POST_KERNEL_CHECK;

	  }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);
}  // namespace caffe
