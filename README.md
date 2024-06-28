# DeepMachining
This is the official release code of the paper: "DeepMachining: Online Prediction of Machining Errors of Lathe Machines"
# Data
We released some sample data from the WC_TAN-MS and WC_TC-AS datasets to demonstrate how to run the code for model fine-tuning.   
# How to Run  
1. To perform fine-tuning on the **WC_TAN-MS** dataset, run the following command:  
   ```bash  
   python main.py --config_file='configs/config_WC_TAN-MS.yml'  
   ```  
   
2. To perform fine-tuning on the **WC_TC-AS** dataset, run the following command:  
   ```bash  
   python main.py --config_file='configs/config_WC_TC-AS.yml'  
   ```
# Citation
```
@article{lu2024deepmachining,
  title={DeepMachining: Online Prediction of Machining Errors of Lathe Machines},
  author={Lu, Xiang-Li and Hsu, Hwai-Jung and Chou, Che-Wei and Kung, HT and Lee, Chen-Hsin},
  journal={arXiv preprint arXiv:2403.16451},
  year={2024}
}
```
