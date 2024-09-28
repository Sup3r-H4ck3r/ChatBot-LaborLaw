import os
import re
import docx2txt

def get_docs(path, type= '.docx'):
    docs = []
    for file in os.listdir(path):
        if file.endswith(type):
            docs.append(docx2txt.process(os.path.join(path, file)))
    return docs

def processing(document):
    text_splits = []
    try:
        so = re.findall(":\s(\d+\/\d+\/.+)", document)[0]
        year = re.findall(":\s\d+\/(\d+)\/.+", document)[0]
    except IndexError:
        so = re.findall(":\s(\d+\/.+)", document)[0]
        year = ""
    name = re.findall(r'(NGHỊ ĐỊNH|LUẬT|QUYẾT ĐỊNH|THÔNG TƯ|BỘ LUẬT)\s+\n+(.+)', document)
    if name[0][0] == "NGHỊ ĐỊNH":
        name = name[0][0].title() + ' ' + name[0][1].title()+ ' số ' + so
    else:
        name = name[0][0].title() + ' ' + name[0][1].title()+ ' năm ' + year
    name = name.strip()
    chuong = re.findall("Chương\s[IVX]+\n+\s.+", document)
    if chuong != []:
        nd_chuong = re.split("Chương\s[IVX]+\n+\s.+", document)
        for i in range(1, len(nd_chuong)):
            muc = re.findall("Mục\s\d+\.\s.+", nd_chuong[i])
            if muc != []:
                nd_muc = re.split("Mục\s\d+\.\s.+", nd_chuong[i])
                for ii in range(1, len(nd_muc)):
                    dieu = re.findall("Điều\s\d+\.\s.+", nd_muc[ii])
                    nd_dieu = re.split("Điều\s\d+\.\s.+", nd_muc[ii])
                    for iii in range(1, len(nd_dieu)):
                        text_splits.append('\n\n'.join([name, dieu[iii-1].strip(), nd_dieu[iii].strip()]))

            else:
                dieu = re.findall("Điều\s\d+\.\s.+", nd_chuong[i])
                nd_dieu = re.split("Điều\s\d+\.\s.+", nd_chuong[i])
                for ii in range(1, len(nd_dieu)):
                    text_splits.append('\n\n'.join([name, dieu[ii-1].strip(), nd_dieu[ii].strip()]))
    else:
        dieu = re.findall("Điều\s\d+\.\s.+", document)
        nd_dieu = re.split("Điều\s\d+\.\s.+", document)
        for i in range(1, len(nd_dieu)):
            text_splits.append('\n\n'.join([name, dieu[i-1].strip(), nd_dieu[i].strip()]))

    return text_splits


def text_splitter(path, type = '.docx'):
    documents = get_docs(path, type)
    result = []
    for i in documents:
        result.append(processing(i))
    return result