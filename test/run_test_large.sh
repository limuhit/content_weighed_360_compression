
#!/bin/bash
mse=false
if [ "$mse" = true ]; then
echo "testing mse models"
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_mse/dyn_v3_2_best.pth.tar --log results_large/mse_2.txt --gpu-id 0 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_mse/dyn_v3_3_best.pth.tar --log results_large/mse_3.txt --gpu-id 1 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_mse/dyn_v3_4_best.pth.tar --log results_large/mse_4.txt --gpu-id 2 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_mse/dyn_v3_5_best.pth.tar --log results_large/mse_5.txt --gpu-id 3 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_mse/dyn_v3_6_best.pth.tar --log results_large/mse_6.txt --gpu-id 0 &
else
echo "testing ssim models"
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_ssim/dyn_ssim_v3_1_best.pth.tar --log results_large/ssim_1.txt --gpu-id 0 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_ssim/dyn_ssim_v3_2_best.pth.tar --log results_large/ssim_2.txt --gpu-id 1 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_ssim/dyn_ssim_v3_25_best.pth.tar --log results_large/ssim_25.txt --gpu-id 2 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_ssim/dyn_ssim_v3_3_best.pth.tar --log results_large/ssim_3.txt --gpu-id 3 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_ssim/dyn_ssim_v3_4_best.pth.tar --log results_large/ssim_4.txt --gpu-id 0 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_ssim/dyn_ssim_v3_5_best.pth.tar --log results_large/ssim_5.txt --gpu-id 1 &
"/home/csmuli/anaconda3/bin/python" "/home/csmuli/PCONV2/new/test_model_dyn_large.py" --checkpoint dyn_ssim/dyn_ssim_v3_6_best.pth.tar --log results_large/ssim_6.txt --gpu-id 2 &
fi

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi
