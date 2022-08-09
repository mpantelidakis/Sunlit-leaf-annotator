
## Installation

### Windows

 Download and install Anaconda for Python 2.7 64-bit https://www.anaconda.com/distribution/
 
 Open an Anaconda2 terminal
 
 ```
 $ cd path/to/your/project
 ```
 
 ```
 $ pip install -r requirements.txt
 ```

### How to use

Place all images you wish to annotate in the data/train/ directory

Once you are in the project's root folder

```
 $ cd src
```

```
 $ python main.py
```


 Once the UI is initialized go to File -> Open an Image and select one image from the "train" folder
 
  Segmentation -> Execute (You can play around with the different segmentators or configurations to see what happens, but i suggest   using the preconfigured ones)

After the segmentation finishes, you will see that the image is now split and each segment has a border around it.

### Annotation steps

1.  By default all segments are annotated as Noise.
2.  Click on the Sunlit class (on the left of the panel) and annotate the sunlit leaves.

**If you missclick by accident, pick the correct class and click again**

**Finally, if you open an already annotated image, the annotation will be reset, so keep track of your completed images**

You can find the generated mask for your image inside data/train_labels/


Original software found at http://git.inovisao.ucdb.br/inovisao/pynovisao .

NPOSL-30 https://opensource.org/licenses/NPOSL-3.0 - Free for non-profit use (E.g.: Education, scientific research, etc.). Contact Inovisão's Prof. Hemerson Pistori (pistori@ucdb.br), should any interest in commercial exploration of this software arise.

### Demos
* https://www.youtube.com/watch?v=lnoXL1hGTJI
* https://www.youtube.com/watch?v=Q-cjCxUqW_Q
