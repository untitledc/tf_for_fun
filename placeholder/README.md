# TF Dataset API + separated tf.Graph

1. Use Dataset iterator rather than placeholder+feed_dict
    * try_placeholder.py - simple feed_dict mechanism
    * try_no_placeholder.py - use iterator and get_next to feed data
1. Use separated tf.Graph for training and inference
    * separated_graph.py - both use the same iterator protocol
    * separated_graph2.py - use placeholder+feed_dict for inference
