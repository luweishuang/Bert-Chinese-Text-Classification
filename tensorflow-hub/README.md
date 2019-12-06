
下载网页端hub模型到本地：
$ mkdir /tmp/moduleA
# Download the module, and uncompress it to the destination folder. You might want to do this manually.
$ curl -L "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1?tf-hub-format=compressed" | tar -zxvC /tmp/moduleA
# Test to make sure it works.
$ python
> import tensorflow_hub as hub
> hub.Module("/tmp/moduleA")

