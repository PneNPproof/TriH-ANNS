./trih/bin/trih_anns gist-960-euclidean.hdf5 b gist_128 128 0.9
./trih/bin/trih_anns cohere-768-euclidean.hdf5 b cohere_128 128 0.9
./trih/bin/trih_anns openai-1536-euclidean-shuffled.hdf5 b openai_128 128 0.9
./trih/bin/trih_anns sift-128-euclidean-shuffle.hdf5 b sift_64 64 0.9
./trih/bin/trih_anns msong-420-euclidean.hdf5 b msong_128 128 0.9
./trih/bin/trih_anns imagenet-150-euclidean.hdf5 b imagenet_128 128 0.9

./trih/bin/trih_anns gist-960-euclidean.hdf5 s gist_128 1000 300 100 128 4 16 16 0 # 0.9521 0.9504
./trih/bin/trih_anns cohere-768-euclidean.hdf5 s cohere_128 1000 300 100 128 4 16 16 0 # 0.9592 0.9544
./trih/bin/trih_anns openai-1536-euclidean-shuffled.hdf5 s openai_128 1000 15 5 128 4 16 16 0 # 0.8600 0.8573
./trih/bin/trih_anns sift-128-euclidean-shuffle.hdf5 s sift_64 1000 300 100 128 4 16 16 0 # 0.9899 0.9862
./trih/bin/trih_anns msong-420-euclidean.hdf5 s msong_128 1000 300 100 128 4 16 16 0 # 0.9729 0.9611
./trih/bin/trih_anns imagenet-150-euclidean.hdf5 s imagenet_128 1000 300 100 128 4 16 16 0 # 0.9965 0.9943
