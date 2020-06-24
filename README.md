# Image Descriptor

### The goal of the project is to generate dense captions for images by learning visual representations from the rich text descriptions.

## Hypothesis: 
> MS COCO dataset (2014) has 5 captions for each image in the dataset. These are rich dense captions which contains information about the image (descriptive and positional). If we can use this information to learn the visual representations we belive this could perform much better on the downstream tasks

## Dataset
> <a href="https://cocodataset.org/#download">MS COCO - Image Captioning Dataset</a>

## Task List
 - [ ] Reproduce the results of [VirTex: Learning Visual Representations from Textual Annotations](https://arxiv.org/abs/2006.06666)
 - [ ] Create Web App for Inference
 - [ ] Improve Textual Head

## Technology Stack:
> PyTorch

## Install dependencies
``` pip install -r dev-requirements.txt ```

## Contributing ##
### Code Style ###

Code is formatted using the [black](https://github.com/ambv/black) style. Merges to master are prevented if code does not conform.

To apply black to your code, run black from the root Prefect directory:

```
black .
```
Formatting can be easy to forget when developing, so you may choose to install a pre-push hook for black, as follows:

```
pre-commit install --hook-type pre-push
```

Once installed, you won't be allowed to `git push` without passing black.

In addition, a number of extensions are available for popular editors that will automatically apply black to your code.

## Team:
- [Sumanth Doddapaneni](https://www.linkedin.com/in/sumanth-doddapaneni-25494b130/)
- [BSNV Chaitanya](https://www.linkedin.com/in/basava-sai-naga-viswa-chaitanya-665083172/)
- [Rusheel Gollakota](https://www.linkedin.com/in/rusheel-gollakota-028612145/) 
