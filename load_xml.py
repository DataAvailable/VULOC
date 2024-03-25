from xml.dom.minidom import parse
import xml.dom.minidom


# 解析XML文档
def read_xml():
    # 使用minidom解析器打开 XML 文档
    line_dic = {}
    DOMTree = xml.dom.minidom.parse("./manifest.xml")
    collection = DOMTree.documentElement
    if collection.hasAttribute("shelf"):
        print("Root element : %s" % collection.getAttribute("shelf"))
    # 在集合中获取所有testcase
    testcases = collection.getElementsByTagName("testcase")
    for testcase in testcases:  # 读取每一个testcase标签里的内容
        files = testcase.getElementsByTagName('file')
        for i in range(files.length):
            file = files[i]
            filepath = file.getAttribute("path")
            if filepath.endswith(".c"):
                name = filepath.split('.')[0]  # 去除文件名后缀
                # print(name)
                if file.getElementsByTagName("flaw"):  # 读取flaw标签
                    tag = file.getElementsByTagName("flaw")
                    for j in range(tag.length):
                        line = tag[j].getAttribute("line")  # 获取行号
                        line_dic[name] = line
                        # print(line)
    return line_dic  # 返回存放文件名以及相应的漏洞行数(或无漏洞)的字典
