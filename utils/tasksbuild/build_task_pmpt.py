
import json

my_list_gaode_type1 = [

"帮我导航去龙湖闵行天街。",
"帮我规划一条去龙湖上海闵行天街的路线。",
"帮我打开导航，目的地是龙湖闵行天街。", 
"请带我去龙湖上海闵行天街。",
"导航一下到龙湖闵行天街。", 
"我要去龙湖上海闵行天街，帮我导航。",
"请设置目的地为龙湖闵行天街。", 
"开导航，去龙湖上海闵行天街。",
"给我指路去龙湖闵行天街。", 
"帮我查一下去龙湖上海闵行天街的路线。",
"Navigate to Longfor Shanghai Minhang Paradise Walk.",
"Plan a route to Longfor Shanghai Minhang Paradise Walk.",
"Open navigation to Longfor Shanghai Minhang Paradise Walk.",
"Take me to Longfor Shanghai Minhang Paradise Walk.",
]



my_list_gaode_type2 = [

]

meituan_type1 = [
    "帮我搜索一下三米粥铺。",

    "查找一下三米粥铺。",

    "帮我找找三米粥铺在哪。",

    "搜一下三米粥铺。",

    "我想找三米粥铺，帮我查查。",

    "帮我看看附近有没有三米粥铺。",

    "查一下三米粥铺的位置。",

    "找一下最近的三米粥铺。",

    "搜索附近的三米粥铺。",

    "帮我定位三米粥铺。"
]


gaode_type2 = [
    "帮我打车去大零号湾。",

"从默认位置出发，打车到大零号湾。",

"叫辆车，从默认地点到大零号湾。",

"帮我叫车去大零号湾。",

"我想打车去大零号湾。",

"帮我预约一辆车去大零号湾。",

"打开打车服务，目的地是大零号湾。",

"帮我叫辆车，从默认出发地去大零号湾。",

"打车去大零号湾，从默认出发点走。",

"请帮我安排一辆车前往大零号湾。"
]

wyy_task = [
    "打开网易云音乐，查找邓紫棋。",

    "在网易云上搜一下邓紫棋的歌曲。",

    "去网易云音乐里找找邓紫棋。",

    "搜索歌手邓紫棋（网易云）。",

    "在网易云音乐APP里输入“邓紫棋”进行搜索。",

    "打开NetEase Cloud Music，search G.E.M.。",

    "在网易云里找一下邓紫棋的歌单。",

    "在网易云音乐搜索栏输入邓紫棋。",

    "用网易云查找G.E.M.的歌曲。",

    "在网易云音乐上搜G.E.M.邓紫棋的作品。"
]

JD_data = [
    "请帮我查找一下影石Insta360 Go3S这款拇指相机.",
    "搜索影石Insta360 Go3S 拇指相机.",
    "我想搜索“影石Insta360 Go3S”这个产品。"
    "我想在京东了解一下Insta360 Go3S拇指相机，帮我搜索下"
]

JD_3data = [
    "帮我将影石Insta360 Go3S这款拇指相机 加入购物车",
    "将影石Insta360 Go3S这款拇指相机 加入购物车"
]

JD_4 = ["进入影石Insta360店铺，帮我搜索instago360ß"]


B2 = [
    "在B站搜索starcraft并播放第一个视频",
    "于B站查找starcraft然后点开首个结果",
    "在B站上搜寻starcraft并打开排第一的视频",
    "到B站查询starcraft并播放在最前面的视频",
    "在B站输入starcraft进行搜索并点击首个视频播放",
    "通过B站搜索starcraft然后观看第一条",
    "使用B站的搜索功能找starcraft并播放首条内容",
    "在B站检索starcraft并运行第一个结果",
    "从B站搜索starcraft然后开启第一个视频",
    "在B站寻找starcraft并开始播放找到的第一个视频"
]
# 写入文件

B3= [
    "在B站查找一下UP主小潮院长",
    "于B站搜索一下UP主小潮院长",
    "去B站搜一下小潮院长这个UP主",
    "在B站上查询一下UP主小潮院长",
    "到B站去搜索UP主小潮院长",
    "用B站搜一下UP主小潮院长",
    "在B站里找一下小潮院长的频道",
    "打开B站搜索UP主小潮院长",
    "从B站搜一下小潮院长",
    "在B站检索一下UP主小潮院长"
]

B9 = [
    "在UP主小潮院长的垃圾人视频下面评论一个赞",
    "到小潮院长的垃圾人视频下方发表评论点赞",
    "给小潮院长的垃圾人视频留言点赞",
    "在小潮院长的垃圾人视频下回复赞",
    "去UP主小潮院长的垃圾人视频下面评论发赞",
    "在小潮院长垃圾人视频的评论区写个赞",
    "找到小潮院长的垃圾人视频并在下面评论赞",
    "于小潮院长垃圾人视频的留言区评论赞",
    "在小潮院长的垃圾人视频下方用评论表达赞",
    "点开小潮院长的垃圾人视频并在下面评论赞"
]
path = "/Users/fff/Desktop/mobiagent/MobiBench/data/rawdata/bilibili/type9/task.json"

with open(path, "w", encoding="utf-8") as f:
    json.dump(B9,f,ensure_ascii=False, indent=2)