with open('./acc1.txt','r') as f:
    acc = f.readlines()

{'class_id': 11, 'class_name': 'writing', 'samples_num': 315.0, 'class_accuracy': 0.6095238095238096, 'top_5': [['writing', 0.6095238095238096], ['typing on a keyboard', 0.19682539682539682], ['reading', 0.11428571428571428], ['playing with phone/tablet', 0.047619047619047616], ['clapping', 0.009523809523809525]]}

import re
import numpy as np
#易混淆阈值达到0.02及以上
def construct_dict_from_string(s):
    ans = {}
    pattern = re.compile("(?<='class_name': ).+?(?=')")
    ans['class_name'] = re.findall(pattern,s)[0][1:]
    # print(ans['class_name'])
    pattern = re.compile("(?<='top_5': ).+(?=})")
    s1 = re.findall(pattern,s)[0]
    # get_confusing_action(s1,0.02)
    # ans['top'] = re.findall(pattern,s)[0][1:]

class_relation = np.zeros((60,60))
def get_confusing_action(current_classid,s,score):
    """
    score：阈值，找到混淆程度大于该阈值的动作
    """
    s = "[['kicking other person', 0.9554140127388535], ['kicking something', 0.022292993630573247], ['punching/slapping other person', 0.009554140127388535], ['pushing other person', 0.006369426751592357], ['walking apart from each other', 0.0031847133757961785]]"
    pattern = re.compile("\[.+?\]")
    ans = re.findall(pattern,s)
    for i in ans:
        pattern = re.compile('\[+(.+),(.+)\]')
        action = re.findall(pattern,i)[0][0]
        action_id = get_id_by_action_name(action)
        confusion_rate = re.findall(pattern,i)[0][1]

    # print(ans)
get_confusing_action("",0.02)


def get_id_by_action_name(action_name):
    for i in range(NTURGBD_CLASS_NAME_ID):
        if action_name == NTURGBD_CLASS_NAME_ID[i]:
            return i


NTURGBD_CLASS_NAME_ID = ['drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 'throw', 'sitting down', 'standing up (from sitting position)', 'clapping', 'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap', 'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket', 'hopping (one foot jumping)', 'jump up', 'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard', 'pointing to something with finger', 'taking a selfie', 'check time (from watch)', 'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute', 'put the palms together', 'cross hands in front (say stop)', 'sneeze/cough', 'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache/heart pain)', 'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition', 'use a fan (with hand or paper)/feeling warm', 'punching/slapping other person', 'kicking other person', 'pushing other person', 'pat on back of other person', 'point finger at the other person', 'hugging other person', 'giving something to other person', "touch other person's pocket", 'handshaking', 'walking towards each other', 'walking apart from each other', 'put on headphone', 'take off headphone', 'shoot at the basket', 'bounce ball', 'tennis bat swing', 'juggling table tennis balls', 'hush (quite)', 'flick hair', 'thumb up', 'thumb down', 'make ok sign', 'make victory sign', 'staple book', 'counting money', 'cutting nails', 'cutting paper (using scissors)', 'snapping fingers', 'open bottle', 'sniff (smell)', 'squat down', 'toss a coin', 'fold paper', 'ball up paper', 'play magic cube', 'apply cream on face', 'apply cream on hand back', 'put on bag', 'take off bag', 'put something into a bag', 'take something out of a bag', 'open a box', 'move heavy objects', 'shake fist', 'throw up cap/hat', 'hands up (both hands)', 'cross arms', 'arm circles', 'arm swings', 'running on the spot', 'butt kicks (kick backward)', 'cross toe touch', 'side kick', 'yawn', 'stretch oneself', 'blow nose', 'hit other person with something', 'wield knife towards other person', 'knock over other person (hit with body)', 'grab other person’s stuff', 'shoot at other person with a gun', 'step on foot', 'high-five', 'cheers and drink', 'carry something with other person', 'take a photo of other person', 'follow other person', 'whisper in other person’s ear', 'exchange things with other person', 'support somebody with hand', 'finger-guessing game (playing rock-paper-scissors)']