
root_path = "./data"

class  opts_ntu_60_cross_view():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "hidden_size":1024,
    "num_head":1,
    "num_layer":1,
    "num_class":60
    }
  
    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-60-AGCN/xview/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }
   