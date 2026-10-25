# Assignment 2 Report
- **Name**: Andrea Saxod
- **Student ID**: 18996
- **Email**: asaxod.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem and the cost of being wrong
* The decision this model supports is whether the system tells a lifeguard if a swimmer looks like they are drowning. Currently, a lifeguard must decide based on watching the water. This model is meant to assist the lifeguard in their duties.
* The model functions in an indoor pool with four cameras, each in a corner of the pool. Each camera takes one picture every 5 seconds, and the model analyzes each image. If a swimmer appears to be drowning, the system alerts the lifeguard to check and proceed.
* In this model, the positive class will be for "drowning". Each picture is one case, and the pool is open from 8:00 to 18:00. This means the system will review 28,800 pictures a day (4 cameras x 720 pictures per hour x 10 hours).

The two mistakes cost very different amounts: 
* A false alarm: The lifeguard looks at a swimmer who is fine, which takes about 30 seconds. This costs €0.117, because a lifeguard costs the pool around €14.05 an hour.
* A missed drowning: This consists of a picture that shows someone in trouble and nobody is told. I price this at €42.50. A death costs around €170,000 in compensation. After research, I found out that about 0.15% of serious incidents end in a death. In our data, about one incident appears in about 6 pictures, so I split the cost across 6 pictures: 170,000 x 0.0015 / 6 = €42.50.

In short, one miss is about 360 times worse than one false alarm. Drowning, however, is very rare: only about 6 pictures in a million show it. This shapes every choice below: how often it really happens, which score I lead with (drowning recall), where the threshold goes, and how few pictures a person needs to check.


## Dataset
I used a public Roboflow Universe dataset: workspace cuizh, project water-rescue-dataset, version 1 [water-rescue-dataset Model > Overview](https://universe.roboflow.com/cuizh/water-rescue-dataset). This dataset is an object detection dataset, meaning the model draws a box around each person and labels the class.
The data has three classes: drowning, float, and swimmer. One image is a photo or video frame of a water scene, with each person boxed and labeled.
This dataset fits the problem because it includes the class the pool cares about (drowning) alongside people who are fine (floating people and swimmers). The model needs to tell distress apart from normal swimming, and the footage doesn’t look like it comes from a fixed pool camera. The next section explains this in more detail.

## What the dataset's composition implies

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

The data is split into 3,846 training images, 369 validation images, and 181 test images. The training set consists of the original images, each copied three times with a minor change (around 1,282 different images). The validation and test sets have no copies. We stretched every image to 512 by 512.

| Class | Boxes in train / val / test | Test images with it | Typical box size (share of the image) | Where most images come from |
|---|---|---|---|---|
| drowning | 1,623 / 157 / 80 | 80 | 38% | 1,852 of 1,860 from the web or video |
| float | 1,069 / 143 / 44 | 7 | 0.6% (70% of boxes are under 1%) | 1,156 of 1,256 from drones |
| swimmer | 2,951 / 284 / 144 | 94 | 6.6% | 2,627 of 3,379 from a webcam |

I had to guess the image source from the start of each file name, so this is only a rough guide for me.
This means how much I can trust my results:
* ##Each image only has one class.## 

## Finetuning and baselines

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*

## Per-class evaluation and failure modes

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

## Cost matrix and operating point

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

## What a person still has to check

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
