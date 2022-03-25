#!/usr/bin/env python
# coding: utf-8

# In[27]:


from __future__ import print_function


# In[28]:


def colors(path):
    import binascii
    import struct
    from PIL import Image
    import numpy as np
    import scipy
    import scipy.misc
    import scipy.cluster
    from tensorflow.keras.preprocessing import image
    import cv2
    NUM_CLUSTERS = 10
    
    # print('reading image')
    im=cv2.imread(path) # CHANGE THIS
    # im = im.resize((150, 150))   # optional, to reduce time
    im = cv2.resize(im, (150,150), interpolation = cv2.INTER_AREA)
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    # print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    # print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences

    index_max = np.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    # print('most frequent is %s (#%s)' % (peak, colour))
    avg_count = counts/sum(counts)
    avg_count
    zip_iterator = zip(avg_count, codes)
    a_dict = dict(zip_iterator)
    sorted_dict = dict(sorted(a_dict.items(),reverse=True))
    sorted_dict
    
    
    from scipy.spatial import KDTree
    from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

    def convert_rgb_to_names(rgb_tuple):
    
        # a dictionary of all the hex and their respective names in css3
        css3_db = CSS3_HEX_TO_NAMES
        names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))
    
        kdt_db = KDTree(rgb_values)
        distance, index = kdt_db.query(rgb_tuple)
        return f'{names[index]}'
    
    
    closest_colors = []
    for keys,values in sorted_dict.items():
        rgb = [int(values[x]) for x in range(3)]
        # print("Key : {}, RGB : {}, closest match: {}".format(keys, rgb, convert_rgb_to_names(rgb)))
        closest_colors.append(convert_rgb_to_names(rgb))
    colors = [i for n, i in enumerate(closest_colors) if i not in closest_colors[:n]]

    # print(colors)
    # print('done')
    return colors


# In[26]:


