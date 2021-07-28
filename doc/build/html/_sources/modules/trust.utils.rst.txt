Utilities
=========

Custom Datasets
---------------

.. automodule:: trust.utils.custom_dataset
    :members:
    :exclude-members: DuplicateChannels

Other Utilities
---------------

.. automodule:: trust.utils.utils
    :members:

Models
------

We have incorporated several neural network architectures in TRUST. Below is a list of neural network architectures:
 - densenet
 - dla
 - dla_simple
 - dpn
 - efficientnet
 - googlenet
 - lenet
 - mobilenet
 - mobilenetv2
 - pnasnet
 - preact_resnet
 - regnet
 - resnet
 - resnext
 - senet
 - shufflenet
 - shufflenetv2
 - vgg


**To use a custom model architecture, ensure the model architecture has the following:**

The forward method should have two more variables:

#. A boolean variable *last* which -

	*If *true*: returns the model output and the output of the second last layer

	*If *false*: Returns the model output.

#. A boolean variable ‘freeze’ which -

	*If *true*: disables the tracking of any calculations required to later calculate a gradient i.e skips gradient calculation over the weights

	*If *false*: otherwise

#. get_embedding_dim() method which returns the number of hidden units in the last layer.