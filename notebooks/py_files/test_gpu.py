# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: dementia_xai_gpu_py-3.10_tf-2.17.1_cuda-12.3
#     language: python
#     name: dementia_xai_gpu_py-3.10_tf-2.17.1_cuda-12.3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.19
# ---

# %%
import tensorflow as tf
print("tensor-flow-version: ", tf.__version__)
print("#######################")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("#######################")
