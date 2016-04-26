#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  //CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_AP.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_AN.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_NP.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_AP.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_AN.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_NP.Reshape(bottom[0]->num(), 1, 1, 1);
  diff_sq_AP.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_AN.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_NP.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  // vector of ones used to sum along channels
  summer_vec_AP.Reshape(bottom[0]->channels(), 1, 1, 1);
  summer_vec_AN.Reshape(bottom[0]->channels(), 1, 1, 1);
  summer_vec_NP.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
  {
    summer_vec_AP.mutable_cpu_data()[i] = Dtype(1);
	summer_vec_AN.mutable_cpu_data()[i] = Dtype(1);
	summer_vec_NP.mutable_cpu_data()[i] = Dtype(1);
  }
}
////bottom[0] stores the data of positive example, bottom[1] stores the data of anchored example, bottom[2] stores the data of negative example;
template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[1]->cpu_data(),  // A_data
      bottom[0]->cpu_data(),  // P_data
      diff_AP.mutable_cpu_data());  // A_i - P_i;

  caffe_sub(
	  count,
	  bottom[1]->cpu_data(),  // A_data
	  bottom[2]->cpu_data(),  // N_data
	  diff_AN.mutable_cpu_data());  // A_i - N_i;

  caffe_sub(
	  count,
	  bottom[2]->cpu_data(),  // N_data
	  bottom[0]->cpu_data(),  // P_data
	  diff_NP.mutable_cpu_data());  // N_i - P_i;


  const int channels = bottom[0]->channels();
  Dtype alpha_para = this->layer_param_.triplet_loss_param().margin();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_AP.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_AP.cpu_data() + (i*channels), diff_AP.cpu_data() + (i*channels));
	dist_sq_AN.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
		diff_AN.cpu_data() + (i*channels), diff_AN.cpu_data() + (i*channels));
    loss += std::max(dist_sq_AP.cpu_data()[i]-dist_sq_AN.cpu_data()[i]+alpha_para, Dtype(0.0));
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}



template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
		Dtype alpha_para = this->layer_param_.triplet_loss_param().margin();
		for (int i = 0; i < 3; ++i) 
		{
			if (propagate_down[i]) 
			{
				const Dtype sign = (i == 0) ? -1 : 1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] /
					static_cast<Dtype>(bottom[i]->num());
				int num = bottom[i]->num();
				int channels = bottom[i]->channels();
				for (int j = 0; j < num; ++j) 
				{
					Dtype* bout = bottom[i]->mutable_cpu_diff();
					if (dist_sq_AP.cpu_data()[j]-dist_sq_AN.cpu_data()[j]+alpha_para>0) 
					{  // 对于不满足要求的训练样本，需要将损失反传;
						if (i == 0)
						{
							caffe_cpu_axpby(
								channels,
								alpha,
								diff_AP.cpu_data() + (j*channels),
								Dtype(0.0),
								bout + (j*channels));
						}
						else if (i == 1)
						{
							caffe_cpu_axpby(
								channels,
								alpha,
								diff_NP.cpu_data() + (j*channels),
								Dtype(0.0),
								bout + (j*channels));
						}
						else 
						{
							caffe_cpu_axpby(
								channels,
								alpha,
								diff_AN.cpu_data() + (j*channels),
								Dtype(0.0),
								bout + (j*channels));
						}
					} 
					else {          /////对于已经符合要求的训练样本则将底层的损失设为0;
							caffe_set(channels, Dtype(0), bout + (j*channels));
						 }
					}
				}
			}
}




#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
