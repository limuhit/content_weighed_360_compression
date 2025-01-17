Codes for "Learning Content-Weighted Pseudocylindrical Representation for 360-degree Image Compression"

Requirmed packages:
- pytorch
- cv2 (python-opencv)
- numpy
- compressai (1.2.4.dev0)
 
Install:
* python setup.py install
* cd coder & python setup_linux.py install
	
Running the codec for 360-degree images:
* Encoding:
 	* python pseudo_codec.py --enc --img-file image_names.txt --code-file code_names.txt --model-idx 3 --ssim
 	* python pseudo_codec.py --enc --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
* Decoding:
 	* python pseudo_codec.py --dec --out-file decoded_image_names.txt --code-file code_names.txt --model-idx 3 --ssim
 	* python pseudo_codec.py --dec --out-list a_dec.png b_dec.png --code-list code_a code_b --model-idx 3 --ssim
* Testing (Decoding and evaluate the performance):
 	* python pseudo_codec.py --test --img-file source_image_names.txt --code-file code_names.txt --model-idx 3 --ssim
 	* python pseudo_codec.py --test --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
 	* python pseudo_codec.py --test --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
