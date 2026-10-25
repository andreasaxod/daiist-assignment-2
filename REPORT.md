# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Andrea Saxod
- **Student ID**: 18996
- **Email**: asaxod.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong
* The decision this model supports is whether the system tells a lifeguard if a swimmer looks like they are drowning. Currently, a lifeguard must decide based on watching the water. This model is meant to assist the lifeguard in their duties.
* The model functions in an indoor pool with four cameras, each in a corner of the pool. Each camera takes one picture every 5 seconds, and the model analyzes each image. If a swimmer appears to be drowning, the system alerts the lifeguard to check and proceed.
* In this model, the positive class will be for "drowning". Each picture is one case, and the pool is open from 8:00 to 18:00. This means the system will review 28,800 pictures a day (4 cameras x 720 pictures per hour x 10 hours).

The two mistakes cost very different amounts: 
* A false alarm: The lifeguard looks at a swimmer who is fine, which takes about 30 seconds. This costs €0.117, because a lifeguard costs the pool around €14.05 an hour.
* A missed drowning: This consists of a picture that shows someone in trouble and nobody is told. I price this at €42.50. A death costs around €170,000 in compensation. After research, I found out that about 0.15% of serious incidents end in a death. In our data, about one incident appears in about 6 pictures, so I split the cost across 6 pictures: 170,000 x 0.0015 / 6 = €42.50.

In short, one miss is about 360 times worse than one false alarm. Drowning, however, is very rare: only about 6 pictures in a million show it. This shapes every choice below: how often it really happens, which score I lead with (drowning recall), where the threshold goes, and how few pictures a person needs to check.


## Dataset

*Workspace / project / version on Roboflow Universe, and the link. What one
image represents, what the classes are, and why this dataset actually serves
the problem above rather than merely resembling it.*

I used a public Roboflow Universe dataset: workspace cuizh, project water-rescue-dataset, version 1 [water-rescue-dataset Model > Overview](https://universe.roboflow.com/cuizh/water-rescue-dataset).
## What the dataset's composition implies

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

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
