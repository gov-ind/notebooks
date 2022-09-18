#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'notebook')

x_min, x_max, y_min, y_max = -5, 5, -5, 5
N = 100

def draw(x, y, ax, label=None, clear=True):
    if clear:
        ax.clear()
        ax.grid(True)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([x_min, x_max])
        plt.axvline(x=0, c="black")
        plt.axhline(y=0, c="black")
    ax.scatter(x, y, label=label)
    if label is not None:
        plt.legend()

def display_slider(label, value=0):
    slider = widgets.FloatSlider(
        value=value, 
        min=-10, max=10, step=.1,
        description=label,
        continuous_update=True
    )
    slider.observe(update, 'value')
    display(slider)
    return slider


# In[18]:


import numpy as np

x = np.linspace(x_min, x_max, N)

def net(x, w, b):
    return w * x + b
 
fig, ax = plt.subplots(figsize=(x_max, y_max))
draw(x, net(x, w=0, b=0), ax)
    
def update(change):
    draw(x, net(x, w=weight_slider.value, b=bias_slider.value), ax)
    
weight_slider = display_slider('Weight')
bias_slider = display_slider('Bias')


# In[19]:


fig, ax = plt.subplots(figsize=(x_max, y_max))

w_actual = 1
b_actual = 2
y = w_actual * x + b_actual + np.random.normal(0, .5, N)

draw(x, y, ax)
draw(x, net(x, w=0, b=0), ax, clear=False, label='Predicted')

def update(change):
    draw(x, y, ax)
    draw(x, net(x, w=weight_slider.value, b=bias_slider.value), ax, clear=False, label='Predicted')

weight_slider = display_slider('Weight')
bias_slider = display_slider('Bias')


# In[20]:


fig, ax = plt.subplots(figsize=(x_max, y_max))

w1_actual = 0.2
b1_actual = -2
w2_actual = 2
y = w1_actual * x + b1_actual + np.random.normal(0, .5, N)

y[N // 2:] += w2_actual * x[N // 2:]

draw(x, y, ax)
draw(x, net(x, w=0, b=0), ax, clear=False, label='Predicted')

def update(change):
    draw(x, y, ax)
    draw(x, net(x, w=weight_slider.value, b=bias_slider.value), ax, clear=False, label='Predicted')

weight_slider = display_slider('Weight')
bias_slider = display_slider('Bias')


# In[21]:


fig, ax = plt.subplots(figsize=(x_max, y_max))

w1_actual = 0.2
b1_actual = -2
w2_actual = 2
y = w1_actual * x + b1_actual + np.random.normal(0, .5, N)

y[N // 2:] += w2_actual * x[N // 2:]

draw(x, y, ax)
draw(x, net(x, w=w1_actual, b=b1_actual), ax, clear=False, label='Pred 1')
draw(x, net(x, w=w2_actual, b=b1_actual), ax, clear=False, label='Pred 2')


# In[22]:


fig, ax = plt.subplots(figsize=(x_max, y_max))

w1, w2, b1, b2 = .2, 2, -2, -2

draw(x, net(x, w=w1, b=b1), ax, label='Pred 1')
draw(x, net(x, w=w2, b=b2), ax, clear=False, label='Pred 2')
draw(x, net(x, w=(w1 + w2), b=(b1 + b2)), ax, clear=False, label='Pred Sum')

def update(change):
    w1, w2, b1, b2 = weight_slider1.value, weight_slider2.value, bias_slider1.value, bias_slider2.value
    draw(x, net(x, w=w1, b=b1), ax, label='Pred 1')
    draw(x, net(x, w=w2, b=b2), ax, clear=False, label='Pred 2')
    draw(x, net(x, w=(w1 + w2), b=(b1 + b2)), ax, clear=False, label='Pred Sum')
    
weight_slider1 = display_slider('Weight 1', value=w1)
bias_slider1 = display_slider('Bias 1', value=b1)
weight_slider2 = display_slider('Weight 2', value=w2)
bias_slider2 = display_slider('Bias 2', value=b2)


# In[29]:


fig, ax = plt.subplots(figsize=(x_max, y_max))

def net(x, w, b, b_out=0):
    return np.maximum(0, w * x + b) + b_out

w1, w2, b1, b2 = .2, 2, -2, -2
b_out = 0

def _draw(w1, b1, w2, b2, b_out):
    neuron1_output = net(x, w=w1, b=b1, b_out=b_out)
    neuron2_output = net(x, w=w2, b=b2, b_out=b_out)
    draw(x, y, ax)
    draw(x, neuron1_output, ax, clear=False, label='Pred 1')
    draw(x, neuron2_output, ax, clear=False, label='Pred 2')
    draw(x, neuron1_output + neuron2_output, ax, clear=False, label='Pred Sum')

_draw(w1, b1, w2, b2, b_out)
    
def update(change):
    w1, w2, b1, b2 = weight_slider1.value, weight_slider2.value, bias_slider1.value, bias_slider2.value
    b_out = bias_slider3.value
    _draw(w1, b1, w2, b2, b_out)
    
weight_slider1 = display_slider('Weight 1', value=w1)
bias_slider1 = display_slider('Bias 1', value=b1)
weight_slider2 = display_slider('Weight 2', value=w2)
bias_slider2 = display_slider('Bias 2', value=b2)
bias_slider3 = display_slider('Bias Output', value=b_out)


# In[93]:


import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'notebook')

b = 1
x = np.linspace(-100, 100, 1000)
y = np.maximum(0, x + b)
 
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(x, y)
ax.grid(True)
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
plt.axvline(x=0, c="black", label="x=0")
plt.axhline(y=0, c="black", label="y=0")
 
def update(change):
    ax.clear()
    ax.grid(True)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    plt.axvline(x=0, c="black", label="x=0")
    plt.axhline(y=0, c="black", label="y=0")
    ax.scatter(x, np.maximum(0, x * w1_slider.value + b1_slider.value) * w2_slider.value + b2_slider.value)
    #fig.canvas.draw()
        
w1_slider = widgets.FloatSlider(
    value=1, 
    min=-10, max=10, step=.1,
    description='Weight 1',
    continuous_update=True
)
w1_slider.observe(update, 'value')
display(w1_slider)
w2_slider = widgets.FloatSlider(
    value=1, 
    min=-10, max=10, step=.1,
    description='Weight 2',
    continuous_update=True
)
w2_slider.observe(update, 'value')
display(w2_slider)
b1_slider = widgets.FloatSlider(
    value=1, 
    min=-10, max=10, step=.1,
    description='Bias 1',
    continuous_update=True
)
b1_slider.observe(update, 'value')
display(b1_slider)
b2_slider = widgets.FloatSlider(
    value=1, 
    min=-10, max=10, step=.1,
    description='Bias 2',
    continuous_update=True
)
b2_slider.observe(update, 'value')
display(b2_slider)


# In[94]:


class Net:
    def __init__(self):
        
    def forward(self, x)


# In[ ]:





# In[ ]:





# In[ ]:




