#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os
import os.path
import shutil


def writeHtmlFile(html):
    html["html"] = html["html"] + """
</body>
</html>
    """
    htmlFile = open(html["file_name"], 'w')
    print(html["html"], file=htmlFile)
    htmlFile.close()
    for css in html["style_files"]:
        if not os.path.exists(os.path.join(html["dir_name"], css)):
            try:
                shutil.copyfile(os.path.basename(css), os.path.join(html["dir_name"], css))
            except:
                print("WARNING: css was not copied")


def startHtml(file_name, title, styles):
    styleStr = "\n"
    html = dict()
    html["style_files"] = list()
    if styles:
        for style in styles:
            styleStr = styleStr + '<link rel="stylesheet" href="' + style + '">\n'
            html["style_files"].append(style)
    html["file_name"] = file_name
    html["dir_name"] = os.path.dirname(file_name)
    html["title"] = title
    html["html"] = """<!DOCTYPE html>
<html>
<head>""" + styleStr + """</head>
<title>
    """ + html["title"] + """
</title>
<body>
<h1>
    """ + html["title"] + """
</h1>
"""
    return html


def htmlTable(html, columns, rows, text):
    html["html"] = html["html"] + "<table>\n<tr>\n"
    for i in range(columns):
        html["html"] = html["html"] + "<th>" + text[i] + "</th>\n"
    html["html"] = html["html"] + "</tr>\n"
    for j in range(1, rows):
        html["html"] = html["html"] + "<tr>\n"
        for i in range(columns):
            html["html"] = html["html"] + "<td>" + text[j * columns + i] + "</td>\n"
        html["html"] = html["html"] + "</tr>\n"
    html["html"] = html["html"] + "</table>\n"


def htmlRef(text, url):
    return "<a href=" + url + ">" + text + "</a>"


def htmlWrite(html, text):
    html["html"] = html["html"] + text
