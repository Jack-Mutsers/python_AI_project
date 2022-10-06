(venv) PS D:\Users\jmuts\Documenten\school\jaar 4\Semester 7 Artificial intelligence\personal\code\pyImageSearch>  d:; cd 'd:\Users\jmuts\Documenten\school\jaar 4\Semester 7 Artificial intelligence\personal\code\pyImageSearch'; & 'd:\Users\jmuts\Documenten\school\jaar 4\Semester 7 Artificial intelligence\personal\code\pyImageSearch\venv\Scripts\python.exe' 'c:\Users\jmuts\.vscode\extensions\ms-python.python-2022.14.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher' '57796' '--' 'd:\Users\jmuts\Documenten\school\jaar 4\Semester 7 Artificial intelligence\personal\code\pyImageSearch\ocr-keras-tensorflow\train_ocr_model.py' 
2022-09-30 19:01:37.243967: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-09-30 19:01:37.251842: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
[INFO] loading datasets...
[INFO] datasets loaded.
[INFO] compiling model...
2022-09-30 19:03:33.644469: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-09-30 19:03:33.671700: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublas64_11.dll'; dlerror: cublas64_11.dll not found
2022-09-30 19:03:33.698546: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublasLt64_11.dll'; dlerror: cublasLt64_11.dll not found
2022-09-30 19:03:33.729025: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2022-09-30 19:03:33.755629: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2022-09-30 19:03:33.779091: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusolver64_11.dll'; dlerror: cusolver64_11.dll not found
2022-09-30 19:03:33.808189: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusparse64_11.dll'; dlerror: cusparse64_11.dll not found
2022-09-30 19:03:33.835654: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudnn64_8.dll'; dlerror: cudnn64_8.dll not found
2022-09-30 19:03:33.861657: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1934] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2022-09-30 19:03:33.918765: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
[INFO] training network...
Epoch 1/50
2765/2765 [==============================] - 4055s 1s/step - loss: 1.8369 - accuracy: 0.8726 - val_loss: 0.4880 - val_accuracy: 0.9163
Epoch 2/50
2765/2765 [==============================] - 3784s 1s/step - loss: 0.9446 - accuracy: 0.9268 - val_loss: 0.5146 - val_accuracy: 0.8986
Epoch 3/50
2765/2765 [==============================] - 4206s 2s/step - loss: 0.8644 - accuracy: 0.9332 - val_loss: 0.5895 - val_accuracy: 0.8663
Epoch 4/50
2765/2765 [==============================] - 3753s 1s/step - loss: 0.8231 - accuracy: 0.9369 - val_loss: 0.5188 - val_accuracy: 0.9004
Epoch 5/50
2765/2765 [==============================] - 3703s 1s/step - loss: 0.7943 - accuracy: 0.9391 - val_loss: 0.5881 - val_accuracy: 0.8700
Epoch 6/50
2765/2765 [==============================] - 3691s 1s/step - loss: 0.7781 - accuracy: 0.9398 - val_loss: 0.5436 - val_accuracy: 0.8898
Epoch 7/50
2765/2765 [==============================] - 3703s 1s/step - loss: 0.7603 - accuracy: 0.9412 - val_loss: 0.5684 - val_accuracy: 0.8760
Epoch 8/50
2765/2765 [==============================] - 3679s 1s/step - loss: 0.7499 - accuracy: 0.9424 - val_loss: 0.5756 - val_accuracy: 0.8702
Epoch 9/50
2765/2765 [==============================] - 3725s 1s/step - loss: 0.7410 - accuracy: 0.9434 - val_loss: 0.5282 - val_accuracy: 0.8923
Epoch 10/50
2765/2765 [==============================] - 3737s 1s/step - loss: 0.7293 - accuracy: 0.9442 - val_loss: 0.6046 - val_accuracy: 0.8620
Epoch 11/50
2765/2765 [==============================] - 3733s 1s/step - loss: 0.7199 - accuracy: 0.9454 - val_loss: 0.5517 - val_accuracy: 0.8832
Epoch 12/50
2765/2765 [==============================] - 3764s 1s/step - loss: 0.7127 - accuracy: 0.9467 - val_loss: 0.5271 - val_accuracy: 0.8928
Epoch 13/50
2765/2765 [==============================] - 3974s 1s/step - loss: 0.7047 - accuracy: 0.9478 - val_loss: 0.5074 - val_accuracy: 0.9019
Epoch 14/50
2765/2765 [==============================] - 4121s 1s/step - loss: 0.6963 - accuracy: 0.9488 - val_loss: 0.5044 - val_accuracy: 0.9025
Epoch 15/50
2765/2765 [==============================] - 4314s 2s/step - loss: 0.6922 - accuracy: 0.9495 - val_loss: 0.5188 - val_accuracy: 0.8939
Epoch 16/50
2765/2765 [==============================] - 4641s 2s/step - loss: 0.6820 - accuracy: 0.9505 - val_loss: 0.5218 - val_accuracy: 0.8924
Epoch 17/50
2765/2765 [==============================] - 5007s 2s/step - loss: 0.6806 - accuracy: 0.9507 - val_loss: 0.4861 - val_accuracy: 0.9092
Epoch 18/50
2765/2765 [==============================] - 4215s 2s/step - loss: 0.6752 - accuracy: 0.9513 - val_loss: 0.4718 - val_accuracy: 0.9157
Epoch 19/50
2765/2765 [==============================] - 4679s 2s/step - loss: 0.6713 - accuracy: 0.9520 - val_loss: 0.4822 - val_accuracy: 0.9096
Epoch 20/50
2765/2765 [==============================] - 4420s 2s/step - loss: 0.6675 - accuracy: 0.9525 - val_loss: 0.4746 - val_accuracy: 0.9131
Epoch 21/50
2765/2765 [==============================] - 4124s 1s/step - loss: 0.6627 - accuracy: 0.9529 - val_loss: 0.4878 - val_accuracy: 0.9062
Epoch 22/50
2765/2765 [==============================] - 3875s 1s/step - loss: 0.6594 - accuracy: 0.9534 - val_loss: 0.4391 - val_accuracy: 0.9327
Epoch 23/50
2765/2765 [==============================] - 3739s 1s/step - loss: 0.6583 - accuracy: 0.9536 - val_loss: 0.4591 - val_accuracy: 0.9198
Epoch 24/50
2765/2765 [==============================] - 3797s 1s/step - loss: 0.6553 - accuracy: 0.9543 - val_loss: 0.4335 - val_accuracy: 0.9354
Epoch 25/50
2765/2765 [==============================] - 3952s 1s/step - loss: 0.6480 - accuracy: 0.9549 - val_loss: 0.4363 - val_accuracy: 0.9335
Epoch 26/50
2765/2765 [==============================] - 3780s 1s/step - loss: 0.6431 - accuracy: 0.9551 - val_loss: 0.4564 - val_accuracy: 0.9236
Epoch 27/50
2765/2765 [==============================] - 3681s 1s/step - loss: 0.6476 - accuracy: 0.9556 - val_loss: 0.4276 - val_accuracy: 0.9396
Epoch 28/50
2765/2765 [==============================] - 3718s 1s/step - loss: 0.6415 - accuracy: 0.9557 - val_loss: 0.4343 - val_accuracy: 0.9344
Epoch 29/50
2765/2765 [==============================] - 3716s 1s/step - loss: 0.6425 - accuracy: 0.9553 - val_loss: 0.4332 - val_accuracy: 0.9363
Epoch 30/50
2765/2765 [==============================] - 3732s 1s/step - loss: 0.6364 - accuracy: 0.9570 - val_loss: 0.4152 - val_accuracy: 0.9471
Epoch 31/50
2765/2765 [==============================] - 3715s 1s/step - loss: 0.6321 - accuracy: 0.9573 - val_loss: 0.4323 - val_accuracy: 0.9346
Epoch 32/50
2765/2765 [==============================] - 3697s 1s/step - loss: 0.6302 - accuracy: 0.9575 - val_loss: 0.4122 - val_accuracy: 0.9482
Epoch 33/50
2765/2765 [==============================] - 3704s 1s/step - loss: 0.6303 - accuracy: 0.9574 - val_loss: 0.4058 - val_accuracy: 0.9510
Epoch 34/50
2765/2765 [==============================] - 3645s 1s/step - loss: 0.6291 - accuracy: 0.9581 - val_loss: 0.4186 - val_accuracy: 0.9436
Epoch 35/50
2765/2765 [==============================] - 3672s 1s/step - loss: 0.6222 - accuracy: 0.9588 - val_loss: 0.4075 - val_accuracy: 0.9511
Epoch 36/50
2765/2765 [==============================] - 3670s 1s/step - loss: 0.6233 - accuracy: 0.9589 - val_loss: 0.3997 - val_accuracy: 0.9534
Epoch 37/50
2765/2765 [==============================] - 3687s 1s/step - loss: 0.6207 - accuracy: 0.9594 - val_loss: 0.3966 - val_accuracy: 0.9565
Epoch 38/50
2765/2765 [==============================] - 3670s 1s/step - loss: 0.6174 - accuracy: 0.9597 - val_loss: 0.3877 - val_accuracy: 0.9596
Epoch 39/50
2765/2765 [==============================] - 3675s 1s/step - loss: 0.6153 - accuracy: 0.9598 - val_loss: 0.3915 - val_accuracy: 0.9584
Epoch 40/50
2765/2765 [==============================] - 3686s 1s/step - loss: 0.6142 - accuracy: 0.9604 - val_loss: 0.3906 - val_accuracy: 0.9576
Epoch 41/50
2765/2765 [==============================] - 3659s 1s/step - loss: 0.6112 - accuracy: 0.9607 - val_loss: 0.3858 - val_accuracy: 0.9608
Epoch 42/50
2765/2765 [==============================] - 3663s 1s/step - loss: 0.6129 - accuracy: 0.9606 - val_loss: 0.3883 - val_accuracy: 0.9594
Epoch 43/50
2765/2765 [==============================] - 3690s 1s/step - loss: 0.6062 - accuracy: 0.9612 - val_loss: 0.3842 - val_accuracy: 0.9614
Epoch 44/50
2765/2765 [==============================] - 3661s 1s/step - loss: 0.6071 - accuracy: 0.9614 - val_loss: 0.3750 - val_accuracy: 0.9654
Epoch 45/50
2765/2765 [==============================] - 3679s 1s/step - loss: 0.6067 - accuracy: 0.9615 - val_loss: 0.3767 - val_accuracy: 0.9645
Epoch 46/50
2765/2765 [==============================] - 3658s 1s/step - loss: 0.6024 - accuracy: 0.9616 - val_loss: 0.3783 - val_accuracy: 0.9638
Epoch 47/50
2765/2765 [==============================] - 3650s 1s/step - loss: 0.5980 - accuracy: 0.9622 - val_loss: 0.3798 - val_accuracy: 0.9628
Epoch 48/50
2765/2765 [==============================] - 3679s 1s/step - loss: 0.5971 - accuracy: 0.9624 - val_loss: 0.3715 - val_accuracy: 0.9666
Epoch 49/50
2765/2765 [==============================] - 3646s 1s/step - loss: 0.5975 - accuracy: 0.9625 - val_loss: 0.3684 - val_accuracy: 0.9679
Epoch 50/50
2765/2765 [==============================] - 3657s 1s/step - loss: 0.5972 - accuracy: 0.9623 - val_loss: 0.3735 - val_accuracy: 0.9659
[INFO] evaluating network...
692/692 [==============================] - 187s 269ms/step
              precision    recall  f1-score   support

           0       0.60      0.69      0.64      1381
           1       0.98      0.99      0.98      1575
           2       0.92      0.97      0.94      1398
           3       0.98      0.99      0.99      1428
           4       0.92      0.98      0.95      1365
           5       0.64      0.95      0.77      1263
           6       0.96      0.98      0.97      1375
           7       0.97      0.99      0.98      1459
           8       0.96      0.99      0.98      1365
           9       0.99      0.99      0.99      1392
           A       0.99      1.00      1.00      2774
           B       0.99      0.99      0.99      1734
           C       0.99      0.99      0.99      4682
           D       0.93      0.98      0.95      2027
           E       0.99      0.99      0.99      2288
           F       0.96      1.00      0.98       232
           G       0.98      0.95      0.96      1152
           H       0.98      0.98      0.98      1444
           I       0.98      0.99      0.98       224
           J       0.98      0.97      0.98      1699
           K       0.98      0.98      0.98      1121
           L       0.98      0.98      0.98      2317
           M       0.99      0.99      0.99      2467
           N       0.99      0.99      0.99      3802
           O       0.96      0.93      0.95     11565
           P       1.00      0.99      0.99      3868
           Q       0.97      0.99      0.98      1162
           R       0.99      0.99      0.99      2313
           S       0.99      0.93      0.96      9684
           T       1.00      0.99      0.99      4499
           U       0.99      0.98      0.99      5802
           V       0.96      1.00      0.98       836
           W       0.99      0.99      0.99      2157
           X       0.98      1.00      0.99      1254
           Y       0.99      0.95      0.97      2172
           Z       0.96      0.92      0.94      1215

    accuracy                           0.97     88491
   macro avg       0.96      0.97      0.96     88491
weighted avg       0.97      0.97      0.97     88491

[INFO] serializing network...
1/1 [==============================] - 0s 63ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 31ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 31ms/step
1/1 [==============================] - 0s 17ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 18ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 31ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 30ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 31ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 32ms/step