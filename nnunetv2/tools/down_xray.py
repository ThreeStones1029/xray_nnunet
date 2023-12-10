'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-12-04 21:12:11
LastEditors: ShuaiLei
LastEditTime: 2023-12-04 21:54:45
'''
import urllib.request  # url request
import os              # dirs
 
# pull request
headers = {'111', '666'}
opener = urllib.request.build_opener()
opener.addheaders = [headers]


def readContent(opener, url):
    content = opener.open(url).read().decode('utf8')
    lis = []
    for i in range(len(content)):
        if content[i] == '<':
            if content[i:i+8] == '</A><br>':
                j = i
                while content[j-1] != '>':
                    j = j-1
                if content[j] != '[':
                    lis.append(content[j:i])
    return lis

def downloadOrEnter(name):
    # 判断是否要下载还是进入链接
    # 需要下载的文件类型有：'.csv', '.jpg', '.txt'
    if name[-4:] == '.csv' or name[-4:] == '.jpg' or '.txt':
        return 1
    else:
        return 0

def process(opener, url, loc):
    lis = readContent(opener, url)
    if len(lis) == 1 and downloadOrEnter(lis[0]):
    	# 如果只读取到一个连接，并且这个链接是的类型为目标文件
    	# 则只需要进行下载操作
        urllib.request.urlretrieve(url + '/' + '%20'.join(lis[0].split()), os.path.join(loc, lis[0]))
    else:
        # 除此之外，可能需要同时进行文件夹创建以及文件下载
        for i in lis:
            if downloadOrEnter(i):
                #下载文件到当前目录
                urlnew = url + '/' + '%20'.join(i.split())
                urllib.request.urlretrieve(urlnew, os.path.join(loc, i))
                print(urlnew) # 打印已经下载的文件链接，同时显示进度
            else:
                # 在当前目录创建新的文件夹
                urlnew - url + '/' + '%20'.join(i.split())
                locnew = loc + '/' + i
                os.makedirs(locnew[2:])
                # 递归调用，直到完成全部下载任务
                process(opener, urlnew, locnew)


url = 'https://data.lhncbc.nlm.nih.gov/public/NHANES/X-rays/index.html'
loc = '/home/jjf/Downloads/NHANES'
os.makedirs('data')

process(opener, url, loc)
