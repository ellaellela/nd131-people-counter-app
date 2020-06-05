# Project Write-Up


## Explaining Custom Layers

The process behind converting custom layers involves... Some of the potential reasons for handling custom layers are...
Openvino's MO will recognise the most commonly used NN layers. And for most of them there are readily available optimizations and internal representations (in other words, they can be automatically converted): these layers which have those are called supported layers. Other layers which do not have readily available optimization and/or internal reprsentation are called custom layers. The user must tell openvino what to do with these. One of the ways how to deal with custom layers is to offload everything to the original framework and run it there and then load it back into openvino. The other way is to register the custom layer as an extension to the MO.

## Comparing Model Performance

There is an additional helper file that I have used to run the original model with TF and opencv2 (before the conversion to the IR for openvino): main_cv.py
This helper file I have modelled with the help of the following example on how to use TF and opencv2 together for object detection: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
I have used it only to get the performance information of the original TF model before conversion with MO. This is how it is called:
python main_cv.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssdlite_mobilenet_v2_coco_2018_05_09/
(default -pt is set as 0.5 unless other number has been provided; the frozen_inference_model.pb is assumed to be in the path provided with -m)

For the main.py file I have used parts of my code that I did for the last exercise of the Intel Edge AI Scholarship Foundation course that I had finished in February 2020 prior to taking this course (it is my own work after all).
I have compared main_cv.py results with the output of the following command for the converted IR:
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5

I have merged the video outputs of both scripts into one video using ffmpeg: output_combined.mp4 .
On the left, it is the original model's output, while on the right you can see the converted model's output.


My method(s) to compare models before and after conversion to Intermediate Representations were...
1) I counted the number of frames in which we detected a person with score above the threshold.
2) I did the visual inspection of detections by playing videos one next to each other.
3) I timed and summed the time neeeded for the inference on each frame.
4) I have checked the size of the .pb and .bin (+ .xml) files.

The difference between model accuracy pre- and post-conversion was...
there was no difference in performance. I have measured the number of frames for which we detected a person (with desired accuracy) and the numbers are the same for pre and post-conversion to IR.
It might be possible that conversion to IR did not significantly alter the results of basic operations perfored with weights and inputs (e.g. linear operations fusing and grouped convolution fusing). Also, the precision of IR was left at default FP32, so we did not hamper the accuracy of the pretrained weights.

The size of the model pre- and post-conversion was...
The size of .pb file was approximately 20MB, while the size of the combined .bin and .xml file was approximately 18MB. It is 10% down from the original size.  One needs to notice that the model itself was already very small to begin with.

The inference time of the model pre- and post-conversion was...
The inference time needed to run through all the videos of the original model was 148.24s, while the inference time for all the frames of the model after conversion was 44.45s. The speed-up of 3.3 times is significant. Although, one needs to take into account that the original model was not optimized for operations on Intel's hardware (although, TF used AVX instructions, it is possible that the operation fusing performed during the conversion made even better use of the cache).

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1) Stores/hospitals/pharmacies/galleries
It is hard to miss the current situation with the Covid-19 virus: many restrictions exist for public places and it seems that the situation won't be improving for the foreseeable future. For example, larger supermarkets can let in only a limited number of people depending on the size of the store. I have seen that some employ several people to clean the cart from the leaving customers and give a cart to the entering customer. They count the number of people already inside the store by the number of carts that are there. I found it a bit amusing for being required to take a cart even though I was only buying a loaf of bread. But then I saw a mother and her little kid: the kid was pushing his own empty cart as well.
A more developed version of the app would be able to count the number of people who entered and the number of people who left (tracking should be utilised here) and it would be possible to show the number of available places in the store on a screen outside right next to the opening hours information. This way the stores would save money and the person who is handling the cars would be able to preserve his/her health by not being in the direct contact with all the customers.
Same thing applies to art galleries, bars, pharmacies (although in the case of pharmacies and post offices, it should be noted that they depend on the number of cashiers as well).

2) public gatherings/stadiums
One of the unfortunate dangers of public gatherings in the confined spaces is the panic run and overcrowdines. In some countries the organisers sell more tickets than there is available standing places at (sport) stadiums and concert halls. The stampede that might happen in such cases can end up in human casualties. Even when there is no stampede, just the fact that a lof of people are leaning forward can press the front rows against the walls/bars, which can suffocate the unfortunate ones.
A more developed version of this app might count the number of people who have entered the confined space and limit the new entrances when the safety limit has been reached. It would increase the safety of the participants of public gathering.

3) long-distance buses/aeroplanes
One of the first things we encounter when we board the plane is the air stewardess which greets us and increases the counter in her hand to check on how many people have boarded. A similar case is in the long-distance bus travels once a toilette/cigarete break was made: the driver needs to count the number of people in the bus and wait before leaving if someone is missing.
A more developed version of the app would count the number of people who leave the bus and the number of people who board it. This way, the driver/air stewardess would not need to count the passengers.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
Well, there are many things that have an affect on the edge model. Soem of them are:

1) The effect of the training set
If one looks at the COCO dataset and the corresponding article (the one used for the training of the chosen model) one can notice that the mayority of person images have people facing front (not their back) and that they mostly have a completely different of the picture (in the test-video, the camera is looking from a higher point).

2) The lighting and the background
In darker situations, it will be harder for the model to extract the useful features that are used for the detection. Even if we get a bounding-box, we might still end up with the lower score.

3) The camera angle (the pitch)
As we mentioned int he effect of the training set, the pitch of the camera indirectly leads to model performance. Imagine a situation where our person is standing and we are moving the camera pitch: if we were too high, we would not be able to detect the persons closer to the camera, but we would be able to detect the person further away (would the model detect it is a person if it can't see the lower part of the person's body?). Also, vice versa: if we have a low pitch angle, then the persons further awayy would not be detectable, while the ones closer to the camera would be detected (would the model detect it is a person if it can't see the person's head?)

4) Model accuracy
When we change the weight's accuracy or fuse some layers together, we are risking losing some fo the information encoded in the weights. Depending on the sensitivity of the activation functions, the numerical rounding might lead to accuracy degradation.

Some ways to mitigate this would be to perform some sort of image pre-processing. For example, some of the classical computer vision techniques would help with the lighting and background scenarios, and the camera parameters. Doing a linear transformation of the image (to correct for the fixed camera pitch angle) migth also help, but one needs to be sure that those operations are also reflected in the bounding boxes for the output.

## Model Research

I will briefly explain here how I converted the model. And I will try to explain why I have picked this particular model.

I have first downloaded the model from: http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
which was not listed here on the list of available pretrained models in openvino: https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models

I used the following command to convert to IR (inside the model's untared folder):
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

Now, why did I choose this small model?
Well, the term "on the edge" means that the deployment of the model might be in a very restrictive environment: real-time application on a device with limited amount of memory. This is what I had on my mind when I was looking at the model list on the TF github page. By choosing a "lightweight" model, we can make sure we don't use too much memory for weights (and percentage-wise, more of it fits into RAM), and we are making sure the inference time is lower. Of course, bigger and more complicated models usually give superior detection performance, but it depends form application to application what are the "hard constraints" (real-time or accuracy).
