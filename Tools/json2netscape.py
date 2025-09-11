'''
网页的cookie文件转换为Netscape格式的脚本
'''

import json

def json_to_netscape(json_file, txt_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        cookies = json.load(f)

    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("# Netscape HTTP Cookie File\n")
        for cookie in cookies:
            domain = cookie.get("domain", "")
            include_subdomain = "TRUE" if domain.startswith('.') else "FALSE"
            path = cookie.get("path", "/")
            secure = "TRUE" if cookie.get("secure", False) else "FALSE"
            expiry = int(cookie.get("expirationDate", 9999999999))
            name = cookie.get("name", "")
            value = cookie.get("value", "")
            line = f"{domain}\t{include_subdomain}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n"
            f.write(line)

json_to_netscape("cookies.json", "cookies.txt")
