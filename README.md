# Dual-Branch-Image-Quality-Assessment
The official repository for the Dual-Branch-Image-Quality-Assessment (DBIQA). DBIQA is a full-reference image quality assessment model aiming to discern appealing content deformations. It tackles this issue by comparing deep network features in the kernel representation dissimilarity analysis (KRDA) method. 

The key difference between DBIQA and existing methods is that DBIQA combines two feature comparison approaches. The first is the feature self-similarity comparison, which considers the joint degradation of deep network features and mainly introduces the robustness of content deformations. The second is the feature pairwise comparison branch, which compares deep features from the reference image and the different image pairwise and mainly introduces the differentiability of the model in guiding perceptual image enhancement.

## Advantages of DBIQA:
1.  It not only performs well on **classic synthetic distortion-based datasets** but also on several **image processing algorithm-based** datasets like texture synthesis, deep network-based compression, generative network-based superresolution, etc.

    There are 10 tested datasets:

    &ensp;&ensp;&ensp;(1). Four classic IQA datasets: [TID2013](https://www.sciencedirect.com/science/article/pii/S0923596514001490), [LIVE](https://live.ece.utexas.edu/research/Quality/subjective.htm), [CSIQ](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-19/issue-1/011006/Most-apparent-distortion--full-reference-image-quality-assessment-and/10.1117/1.3267105.short#_=_), [KADID-10k](https://database.mmsp-kn.de/kadid-10k-database.html)

    &ensp;&ensp;&ensp;(2). Two texture-synthesized datasets: [SynTEX](https://asu.elsevierpure.com/en/publications/the-effect-of-texture-granularity-on-texture-synthesis-quality/fingerprints/), and [TQD](https://arxiv.org/abs/2004.07728)

    &ensp;&ensp;&ensp;(3). The official [PIPAL](https://github.com/HaomingCai/PIPAL-dataset) training set and [PIPAL](https://github.com/HaomingCai/PIPAL-dataset) test set, noted as 'PIPAL(train)' and 'PIPAL(test)'. PIPAL(test) can only be verified in the [Colab website](https://codalab.lisn.upsaclay.fr/competitions/1567#participate-submit_results).

    &ensp;&ensp;&ensp;(4). [The Quality Assessment of End-to-End Learned Image Compression dataset](https://dl.acm.org/doi/abs/10.1145/3474085.3475569?casa_token=tjAXmKbOPQkAAAAA:gYRbZ4KIMdxekwZA60EhTWGjuO4R-PHuCiv3WpYrOC4A0N_Q10RxA2uht4gg_V48aQ01jFEWk8xX)

    &ensp;&ensp;&ensp;(5). The Visual Quality Assessment for Super-Resolved Images ([QADS](https://ieeexplore.ieee.org/document/8640853))

    &ensp;&ensp;&ensp;(6). The Screen-Content Image Database ([SCID](https://ieeexplore.ieee.org/document/8266580))

    To better validate and reproduce our results, tested results are all provided in the **results** folder with each compared image name.
    
3.  Differentiability in guiding perceptual image enhancement.

4.  Adaptivity to diverse network architectures, including the VGG, ResNet, SqueezeNet, MobileNet, and EfficientNet.

-----------------------------
## Updating log:
2024/7/30: the repository is created, and the quality assessment result and the image-to-image enhancement results are uploaded in the 'results' folder. 

-----------------------------
## Requirements:
  imageio==2.27.0
  
  matplotlib==3.3.4
  
  numpy==1.23.5
  
  Pillow==10.4.0
  
  torch==1.13.1+cu117
  
----------------------------
#### For Quality Assessment:
    if __name__ == '__main__':
        from PIL import Image
        import argparse
        from utils import prepare_image224
    
        parser = argparse.ArgumentParser()
        parser.add_argument('--ref', type=str, default='images/I07.png')
        parser.add_argument('--dist', type=str, default='images/I07_09_04.png')
        args = parser.parse_args()
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        ref = prepare_image224(Image.open(args.ref).convert("RGB"), resize=True).to(device)
        dist = prepare_image224(Image.open(args.dist).convert("RGB"), resize=True).to(device)
    
        model = DBIQA().to(device)
        model.load_state_dict(torch.load(
            './weights/nobias5_subMean_VGG_PLCC_round4_epoch1.pth',map_location=device))
        model = model.eval()
    
        net   = VGG().to(device)
        net.eval()
        ref_stage = net(ref)
        dist_stage = net(dist)
    
        score = model(ref_stage, dist_stage, as_loss=False) 
        print('score: %.4f' % score.item())

#### For Perceptual image enhancement: Please Refer to 'recover.py'

----------------------------
## Citation
