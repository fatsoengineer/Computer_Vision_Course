I have taken the COCO format json.

Out of all the attributes in the json, we mainly require three attributes from it.
* images
> Each element in the list points to a dataset record. It contains size the name of the image ```file_name```, assigned an unique identifier ```id```, and the size of the image as ```width``` and ```height```

* categories
> ```category``` contains the class tagged as ```supercategory```  and the value of the class as ```name``` and an unique identifier for each combination of ```supercategory``` and its value to identify what the bbox is signifying

 annotations
> * ```annotation``` contains a record for every bbox tagged. each bbox is identified as ```id```. 
> * The attribure ```image_id``` points to the unique dataset image from the ```images``` attribute.
> * Similarly ```category_id``` refers to the ```supercategory```unique value i.e. ```id``` from categories attribute.
> * bbox is the cooridinated of the bbox in sequence of coordinates ```x```, ```y```, ```width```, ```height```.
> * ```segmentation```  attribure contains the ccordinate of all the edges of bbox tagged in which every coordinates are in pair as ```x```, ```y```. 