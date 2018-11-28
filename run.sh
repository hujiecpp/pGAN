python train.py --dataset CUHKStudent --nEpochs 20 --cuda
python test.py --train_data CUHKStudent --test_data CUHKStudent --G1_model G_1_model_epoch_20.pth --G2_model G_2_model_epoch_20.pth --my_layer_model my_layer_model_epoch_20.pth --cuda
python test.py --train_data CUHKStudent --test_data XM2VTS --G1_model G_1_model_epoch_20.pth --G2_model G_2_model_epoch_20.pth --my_layer_model my_layer_model_epoch_20.pth --cuda
python test.py --train_data CUHKStudent --test_data AR --G1_model G_1_model_epoch_20.pth --G2_model G_2_model_epoch_20.pth --my_layer_model my_layer_model_epoch_20.pth --cuda
python test.py --train_data CUHKStudent --test_data light --G1_model G_1_model_epoch_20.pth --G2_model G_2_model_epoch_20.pth --my_layer_model my_layer_model_epoch_20.pth --cuda
python test.py --train_data CUHKStudent --test_data pose --G1_model G_1_model_epoch_20.pth --G2_model G_2_model_epoch_20.pth --my_layer_model my_layer_model_epoch_20.pth --cuda
python test.py --train_data CUHKStudent --test_data celebrity --G1_model G_1_model_epoch_20.pth --G2_model G_2_model_epoch_20.pth --my_layer_model my_layer_model_epoch_20.pth --cuda
