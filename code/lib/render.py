import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

def render_as_image(env,step=0):
    plt.imshow(env.render('rgb_array'))
    plt.text(0,0,str(step),fontsize=16) 
    plt.axis('off')
    plt.show()
    clear_output(wait=True)   


import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
def show_video(path="",name_prefix='cart_pole'):
  mp4list = glob.glob(path+'video/'+name_prefix+'*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    print(mp4)
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")

from gym.wrappers import RecordVideo  ## Moniter<=0.21.0 
def wrap_env(env,path="",name_prefix='cart_pole'):
  env = RecordVideo(env, video_folder=path+'video', video_length=500, name_prefix=name_prefix)
  return env    