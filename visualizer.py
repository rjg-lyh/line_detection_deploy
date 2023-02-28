from visdom import Visdom
import torch
import numpy as np
import time
import json 

class Visualizer(object):
    """ Visualizer
    """
    def __init__(self, port='13579', env='main', id=None):
        self.cur_win = {}
        self.vis = Visdom(port=port, env=env)
        self.id = id
        self.env = env
        # Restore
        ori_win = self.vis.get_window_data()
        ori_win = json.loads(ori_win)
        #print(ori_win)
        self.cur_win = { v['title']: k for k, v in ori_win.items()  }

    def vis_scalar(self, name, x, y, opts=None):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        if self.id is not None:
            name = "[%s]"%self.id + name
        default_opts = { 'title': name }
        if opts is not None:
            default_opts.update(opts)

        win = self.cur_win.get(name, None)
        if win is not None:
            self.vis.line( X=x, Y=y, opts=default_opts, update='append',win=win )
        else:
            self.cur_win[name] = self.vis.line( X=x, Y=y, opts=default_opts)

    def vis_image(self, name, img, env=None, opts=None):
        """ vis image in visdom
        """
        if env is None:
            env = self.env 
        if self.id is not None:
            name = "[%s]"%self.id + name
        win = self.cur_win.get(name, None)
        default_opts = { 'title': name }
        if opts is not None:
                default_opts.update(opts)
        if win is not None:
            self.vis.image( img=img, win=win, opts=opts, env=env )
        else:
            self.cur_win[name] = self.vis.image( img=img, opts=default_opts, env=env )#不设置win，就默认新建一个窗口
    
    def vis_table(self, name, tbl, opts=None):
        win = self.cur_win.get(name, None)

        tbl_str = "<table width=\"100%\"> "
        tbl_str+="<tr> \
                 <th>Term</th> \
                 <th>Value</th> \
                 </tr>"
        for k, v in tbl.items():
            tbl_str+=  "<tr> \
                       <td>%s</td> \
                       <td>%s</td> \
                       </tr>"%(k, v)

        tbl_str+="</table>"

        default_opts = { 'title': name }
        if opts is not None:
                default_opts.update(opts)
        if win is not None:
            self.vis.text(tbl_str, win=win, opts=default_opts)
            print('exist_window'.center(100,'-'))
        else:
            self.cur_win[name] = self.vis.text(tbl_str, opts=default_opts)


if __name__=='__main__':
    import numpy as np
    vis = Visualizer(port=8097, env='main')
    print(vis.cur_win)
    # for k, v in vis.vis.get_window_data().items():
    #     print('k: ',k,'v: ',v)
    # tbl = {"lr": 214, "momentum": 0.9}
    # vis.vis_table("test_table", tbl)
    # tbl = {"lr": 244444, "momentum": 0.9, "haha": "hoho"}
    # vis.vis_table("test_table", tbl)


    # # 实例化一个窗口
    # wind = Visdom()
    # # 初始化窗口信息
    # wind.line([0.], # Y的第一个点的坐标
	# 	  [0.], # X的第一个点的坐标
	# 	  win = 'train_loss', # 窗口的名称
	# 	  opts = dict(title = 'train_loss') # 图像的标例
    # )
    # # 更新数据
    # for step in range(10):
	# # 随机获取loss,这里只是模拟实现
    #     loss = np.random.randn() * 0.5 + 2
    #     wind.line([loss],[step],win = 'train_loss',update = 'append')
    #     time.sleep(0.5)

    # 实例化窗口
    wind = Visdom()
    #vis=Visdom(env="heat_map")

    # # 初始化窗口参数
    # wind.line([[0.,0.]],[0.],win = 'train',opts = dict(title = 'loss&acc',legend = ['loss','acc']))
    # #更新窗口数据
    # for step in range(10):
    #     loss = 0.2 * np.random.randn() + 1
    #     acc = 0.1 * np.random.randn() + 0.5
    #     wind.line([[loss, acc]],[step],win = 'train',update = 'append')
        #time.sleep(2)

    # wind.heatmap(np.array([[240]*6 for _ in range(6)],dtype=np.uint8),win='image')
    # wind.close('train')
    a=np.concatenate((np.arange(0,255),np.arange(255,0,-1)))
    b=np.concatenate((np.arange(0,255),np.arange(255,0,-1)))
 
    print(wind.heatmap(
        X=np.outer(a,b))) #不设置win，就默认新建一个窗口
    # import cv2
    # cv2.imshow('23',np.outer(a,b).astype(np.uint8))
    # cv2.waitKey()
    #Visdom.save(wind,['main'])
