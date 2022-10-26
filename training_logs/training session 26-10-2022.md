PS D:\Users\jmuts\Documenten\school\jaar 4\Semester 7 Artificial intelligence\personal\code\pyImageSearch> & C:/Users/jmuts/AppData/Local/Programs/Python/Python310/python.exe "d:/Users/jmuts/Documenten/school/jaar 4/Semester 7 Artificial intelligence/personal/code/pyImageSearch/ocr-keras-tensorflow/train_ocr_model.py"
current time: 2022-10-26 08:04:03
run time: 2022-10-26 10:01:03
waiting to start
current time: 2022-10-26 08:34:03
waiting to start
current time: 2022-10-26 09:04:03
waiting to start
current time: 2022-10-26 09:34:03
waiting to start
current time: 2022-10-26 10:04:03
done waiting
cuda_malloc_async
run started at: 2022-10-26 10:04:06
[INFO] loading datasets...
[INFO] datasets loaded.
started training session 1
[INFO] loading existing model...
2022-10-26 10:07:50.949077: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-10-26 10:07:52.243756: I tensorflow/core/common_runtime/gpu/gpu_process_state.cc:222] Using CUDA malloc Async allocator for GPU: 0
2022-10-26 10:07:52.244888: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4626 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1
[INFO] training network...
training started at: 2022-10-26 10:07:53
Epoch 1/50
2022-10-26 10:07:56.923831: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8500
4250/4250 [==============================] - ETA: 0s - loss: 2.4363 - accuracy: 0.88232022-10-26 10:14:26.735872: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 435204096 exceeds 10% of free system memory.
4250/4250 [==============================] - 466s 108ms/step - loss: 2.4363 - accuracy: 0.8823 - val_loss: 0.5187 - val_accuracy: 0.9325
Epoch 2/50
4250/4250 [==============================] - 462s 109ms/step - loss: 2.2737 - accuracy: 0.8996 - val_loss: 0.5287 - val_accuracy: 0.9310
Epoch 3/50
4250/4250 [==============================] - 461s 109ms/step - loss: 2.2161 - accuracy: 0.9051 - val_loss: 0.5231 - val_accuracy: 0.9321
Epoch 4/50
4250/4250 [==============================] - 465s 109ms/step - loss: 2.1733 - accuracy: 0.9088 - val_loss: 0.8352 - val_accuracy: 0.7876
Epoch 5/50
4250/4250 [==============================] - 466s 110ms/step - loss: 2.1468 - accuracy: 0.9109 - val_loss: 0.4927 - val_accuracy: 0.9339
Epoch 6/50
4250/4250 [==============================] - 459s 108ms/step - loss: 2.1137 - accuracy: 0.9115 - val_loss: 0.5474 - val_accuracy: 0.9289
Epoch 7/50
4250/4250 [==============================] - 463s 109ms/step - loss: 2.0962 - accuracy: 0.9128 - val_loss: 0.6123 - val_accuracy: 0.8773
Epoch 8/50
4250/4250 [==============================] - 463s 109ms/step - loss: 2.0765 - accuracy: 0.9141 - val_loss: 0.5126 - val_accuracy: 0.9247
Epoch 9/50
4250/4250 [==============================] - 468s 110ms/step - loss: 2.0542 - accuracy: 0.9148 - val_loss: 0.5051 - val_accuracy: 0.9274
Epoch 10/50
4250/4250 [==============================] - 466s 110ms/step - loss: 2.0473 - accuracy: 0.9149 - val_loss: 0.5070 - val_accuracy: 0.9255
Epoch 11/50
4250/4250 [==============================] - 467s 110ms/step - loss: 2.0334 - accuracy: 0.9160 - val_loss: 0.7495 - val_accuracy: 0.8246
Epoch 12/50
4250/4250 [==============================] - 465s 109ms/step - loss: 2.0230 - accuracy: 0.9159 - val_loss: 0.5412 - val_accuracy: 0.9295
Epoch 13/50
4250/4250 [==============================] - 467s 110ms/step - loss: 2.0110 - accuracy: 0.9169 - val_loss: 1.7893 - val_accuracy: 0.7231
Epoch 14/50
4250/4250 [==============================] - 468s 110ms/step - loss: 2.0082 - accuracy: 0.9170 - val_loss: 0.5135 - val_accuracy: 0.9323
Epoch 15/50
4250/4250 [==============================] - 467s 110ms/step - loss: 1.9996 - accuracy: 0.9173 - val_loss: 0.9900 - val_accuracy: 0.8600
Epoch 16/50
4250/4250 [==============================] - 468s 110ms/step - loss: 1.9870 - accuracy: 0.9178 - val_loss: 0.6935 - val_accuracy: 0.9052
Epoch 17/50
4250/4250 [==============================] - 467s 110ms/step - loss: 1.9840 - accuracy: 0.9181 - val_loss: 0.6429 - val_accuracy: 0.9127
Epoch 18/50
4250/4250 [==============================] - 464s 109ms/step - loss: 1.9796 - accuracy: 0.9180 - val_loss: 0.5169 - val_accuracy: 0.9213
Epoch 19/50
4250/4250 [==============================] - 467s 110ms/step - loss: 1.9703 - accuracy: 0.9185 - val_loss: 0.8327 - val_accuracy: 0.7955
Epoch 20/50
4250/4250 [==============================] - 463s 109ms/step - loss: 1.9582 - accuracy: 0.9186 - val_loss: 0.5574 - val_accuracy: 0.9015
Epoch 21/50
4250/4250 [==============================] - 460s 108ms/step - loss: 1.9567 - accuracy: 0.9186 - val_loss: 0.7278 - val_accuracy: 0.8365
Epoch 22/50
4250/4250 [==============================] - 462s 109ms/step - loss: 1.9515 - accuracy: 0.9196 - val_loss: 0.6968 - val_accuracy: 0.8453
Epoch 23/50
4250/4250 [==============================] - 461s 108ms/step - loss: 1.9460 - accuracy: 0.9199 - val_loss: 0.5157 - val_accuracy: 0.9238
Epoch 24/50
4250/4250 [==============================] - 463s 109ms/step - loss: 1.9384 - accuracy: 0.9194 - val_loss: 0.5302 - val_accuracy: 0.9137
Epoch 25/50
4250/4250 [==============================] - 464s 109ms/step - loss: 1.9344 - accuracy: 0.9199 - val_loss: 0.5159 - val_accuracy: 0.9229
Epoch 26/50
4250/4250 [==============================] - 466s 110ms/step - loss: 1.9304 - accuracy: 0.9197 - val_loss: 0.5955 - val_accuracy: 0.8840
Epoch 27/50
4250/4250 [==============================] - 466s 110ms/step - loss: 1.9273 - accuracy: 0.9200 - val_loss: 0.5846 - val_accuracy: 0.8873
Epoch 28/50
4250/4250 [==============================] - 469s 110ms/step - loss: 1.9212 - accuracy: 0.9203 - val_loss: 0.5603 - val_accuracy: 0.9219
Epoch 29/50
4250/4250 [==============================] - 463s 109ms/step - loss: 1.9102 - accuracy: 0.9208 - val_loss: 0.8278 - val_accuracy: 0.8831
Epoch 30/50
4250/4250 [==============================] - 464s 109ms/step - loss: 1.9141 - accuracy: 0.9204 - val_loss: 0.8300 - val_accuracy: 0.8794
Epoch 31/50
4250/4250 [==============================] - 465s 109ms/step - loss: 1.9150 - accuracy: 0.9208 - val_loss: 0.5285 - val_accuracy: 0.9252
Epoch 32/50
4250/4250 [==============================] - 468s 110ms/step - loss: 1.9059 - accuracy: 0.9206 - val_loss: 0.5611 - val_accuracy: 0.8991
Epoch 33/50
4250/4250 [==============================] - 468s 110ms/step - loss: 1.9090 - accuracy: 0.9207 - val_loss: 0.9934 - val_accuracy: 0.7452
Epoch 34/50
4250/4250 [==============================] - 463s 109ms/step - loss: 1.8940 - accuracy: 0.9214 - val_loss: 0.8358 - val_accuracy: 0.8815
Epoch 35/50
4250/4250 [==============================] - 467s 110ms/step - loss: 1.8980 - accuracy: 0.9208 - val_loss: 0.5665 - val_accuracy: 0.8967
Epoch 36/50
4250/4250 [==============================] - 468s 110ms/step - loss: 1.8881 - accuracy: 0.9221 - val_loss: 0.5325 - val_accuracy: 0.9183
Epoch 37/50
4250/4250 [==============================] - 468s 110ms/step - loss: 1.8894 - accuracy: 0.9212 - val_loss: 1.7750 - val_accuracy: 0.5239
Epoch 38/50
4250/4250 [==============================] - 469s 110ms/step - loss: 1.8908 - accuracy: 0.9216 - val_loss: 0.5559 - val_accuracy: 0.9204
Epoch 39/50
4250/4250 [==============================] - 466s 110ms/step - loss: 1.8836 - accuracy: 0.9217 - val_loss: 0.5508 - val_accuracy: 0.9212
Epoch 40/50
4250/4250 [==============================] - 466s 110ms/step - loss: 1.8901 - accuracy: 0.9216 - val_loss: 0.5469 - val_accuracy: 0.9071
Epoch 41/50
4250/4250 [==============================] - 467s 110ms/step - loss: 1.8902 - accuracy: 0.9213 - val_loss: 0.7682 - val_accuracy: 0.8216
Epoch 42/50
4250/4250 [==============================] - 445s 105ms/step - loss: 1.8742 - accuracy: 0.9221 - val_loss: 1.9645 - val_accuracy: 0.6817
Epoch 43/50
4250/4250 [==============================] - 442s 104ms/step - loss: 1.8782 - accuracy: 0.9221 - val_loss: 0.7587 - val_accuracy: 0.8932
Epoch 44/50
4250/4250 [==============================] - 442s 104ms/step - loss: 1.8690 - accuracy: 0.9221 - val_loss: 0.5577 - val_accuracy: 0.9209
Epoch 45/50
4250/4250 [==============================] - 442s 104ms/step - loss: 1.8617 - accuracy: 0.9229 - val_loss: 0.5910 - val_accuracy: 0.9180
Epoch 46/50
4250/4250 [==============================] - 442s 104ms/step - loss: 1.8560 - accuracy: 0.9222 - val_loss: 0.5368 - val_accuracy: 0.9128
Epoch 47/50
4250/4250 [==============================] - 442s 104ms/step - loss: 1.8654 - accuracy: 0.9221 - val_loss: 0.5939 - val_accuracy: 0.8842
Epoch 48/50
4250/4250 [==============================] - 442s 104ms/step - loss: 1.8528 - accuracy: 0.9226 - val_loss: 0.7561 - val_accuracy: 0.8265
Epoch 49/50
4250/4250 [==============================] - 443s 104ms/step - loss: 1.8468 - accuracy: 0.9230 - val_loss: 0.8925 - val_accuracy: 0.8714
Epoch 50/50
4250/4250 [==============================] - 442s 104ms/step - loss: 1.8483 - accuracy: 0.9230 - val_loss: 0.5326 - val_accuracy: 0.9168
[INFO] serializing network...
[INFO] evaluating network...
2022-10-26 16:32:20.059890: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 435204096 exceeds 10% of free system memory.
1063/1063 [==============================] - 21s 20ms/step
              precision    recall  f1-score   support

           0       0.28      0.78      0.41      1381
           1       0.74      1.00      0.85      1575
           2       0.84      0.98      0.90      1398
           3       0.99      0.97      0.98      1428
           4       0.90      0.96      0.93      1365
           5       0.51      0.95      0.67      1263
           6       0.95      0.98      0.96      1375
           7       0.95      0.99      0.97      1459
           8       0.95      0.98      0.96      1365
           9       0.86      0.99      0.92      1392
           A       1.00      0.99      0.99      3045
           B       0.99      0.97      0.98      2025
           C       0.98      0.96      0.97      5171
           D       0.93      0.97      0.95      2231
           E       0.99      0.98      0.99      2408
           F       0.89      0.98      0.93       780
           G       0.99      0.93      0.96      1425
           H       0.97      0.98      0.98      1617
           I       0.92      0.81      0.86       518
           J       0.96      0.97      0.97      2179
           K       0.95      0.99      0.97      1625
           L       0.97      0.97      0.97      2503
           M       0.94      0.99      0.96      2880
           N       0.99      0.99      0.99      4081
           O       0.97      0.74      0.84     11805
           P       0.98      0.99      0.99      4476
           Q       0.95      0.98      0.96      1492
           R       0.98      0.99      0.99      2497
           S       0.99      0.87      0.93     10167
           T       0.98      0.99      0.99      4760
           U       0.97      0.98      0.97      6306
           V       0.92      0.97      0.95      1278
           W       0.97      0.97      0.97      2717
           X       0.94      0.99      0.97      1774
           Y       0.97      0.95      0.96      2748
           Z       0.93      0.88      0.91      1737
           a       0.97      0.89      0.93       408
           b       1.00      0.94      0.97       388
           c       0.58      0.71      0.64       195
           d       0.97      0.98      0.98       475
           e       0.99      0.97      0.98       567
           f       0.87      0.25      0.39       131
           g       0.91      0.70      0.79       405
           h       0.96      0.97      0.96       511
           i       0.53      0.37      0.43       392
           j       0.82      0.77      0.79       200
           k       0.98      0.52      0.68       183
           l       0.51      0.25      0.34       497
           m       0.82      0.37      0.51       267
           n       0.97      0.91      0.94       394
           o       0.57      0.69      0.62       442
           p       0.81      0.56      0.67        78
           q       0.83      0.64      0.72       358
           r       0.99      0.90      0.94       500
           s       0.46      0.78      0.58       196
           t       0.97      0.89      0.93       426
           u       0.59      0.71      0.64       179
           v       0.77      0.58      0.66       243
           w       0.97      0.25      0.40       124
           x       0.97      0.46      0.62       168
           y       0.82      0.53      0.64       114
           z       0.98      0.38      0.55       164

    accuracy                           0.92    106251
   macro avg       0.88      0.83      0.83    106251
weighted avg       0.94      0.92      0.92    106251

1/1 [==============================] - 0s 172ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 35ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 16ms/step
1/1 [==============================] - 0s 29ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 31ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 16ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 29ms/step
1/1 [==============================] - 0s 33ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 17ms/step
1/1 [==============================] - 0s 32ms/step
run finished at: 2022-10-26 16:32:47
total duration: 6:28:40.591459