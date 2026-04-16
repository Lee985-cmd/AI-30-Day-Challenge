"""
Markdown 转公众号格式工具
将 Markdown 文件转换为适合微信公众号的富文本格式
"""
import re
import sys
from pathlib import Path


class MarkdownToWechat:
    """Markdown 转公众号格式转换器"""
    
    def __init__(self):
        self.code_block_counter = 0
    
    def convert(self, markdown_text: str) -> str:
        """
        转换 Markdown 为公众号格式
        
        Args:
            markdown_text: Markdown 文本
            
        Returns:
            转换后的富文本格式（HTML）
        """
        lines = markdown_text.split('\n')
        result = []
        in_code_block = False
        code_lang = ""
        code_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 处理代码块
            if line.strip().startswith('```'):
                if not in_code_block:
                    # 开始代码块
                    in_code_block = True
                    code_lang = line.strip()[3:].strip()
                    code_lines = []
                else:
                    # 结束代码块
                    in_code_block = False
                    code_html = self._format_code_block(code_lines, code_lang)
                    result.append(code_html)
                    code_lines = []
                i += 1
                continue
            
            if in_code_block:
                code_lines.append(line)
                i += 1
                continue
            
            # 处理标题
            if line.startswith('# '):
                result.append(f'<section style="margin: 20px 0;"><strong style="font-size: 18px; color: #2c3e50;">{line[2:]}</strong></section>')
            elif line.startswith('## '):
                result.append(f'<section style="margin: 15px 0;"><strong style="font-size: 16px; color: #34495e;">{line[3:]}</strong></section>')
            elif line.startswith('### '):
                result.append(f'<section style="margin: 12px 0;"><strong style="font-size: 15px; color: #7f8c8d;">{line[4:]}</strong></section>')
            
            # 处理引用块
            elif line.startswith('> '):
                quote_text = line[2:]
                result.append(f'<section style="background: #f8f9fa; padding: 12px 15px; border-left: 4px solid #3498db; margin: 10px 0; border-radius: 4px;"><span style="color: #555; font-size: 14px;">{quote_text}</span></section>')
            
            # 处理列表项
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                item_text = line.strip()[2:]
                # 处理加粗
                item_text = self._process_bold(item_text)
                item_text = self._process_italic(item_text)
                result.append(f'<section style="margin: 8px 0; padding-left: 20px;"><span style="color: #2c3e50; font-size: 15px;">• {item_text}</span></section>')
            
            # 处理有序列表
            elif re.match(r'^\d+\.\s', line.strip()):
                item_text = re.sub(r'^\d+\.\s', '', line.strip())
                item_text = self._process_bold(item_text)
                item_text = self._process_italic(item_text)
                result.append(f'<section style="margin: 8px 0; padding-left: 20px;"><span style="color: #2c3e50; font-size: 15px;">{item_text}</span></section>')
            
            # 处理分隔线
            elif line.strip() == '---':
                result.append('<hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">')
            
            # 处理空行
            elif line.strip() == '':
                result.append('<br>')
            
            # 处理普通段落
            else:
                # 处理加粗、斜体、链接等
                processed_line = self._process_inline_formatting(line)
                if processed_line.strip():
                    result.append(f'<p style="margin: 10px 0; line-height: 1.8; font-size: 15px; color: #333;">{processed_line}</p>')
            
            i += 1
        
        return '\n'.join(result)
    
    def _process_bold(self, text: str) -> str:
        """处理加粗 **text**"""
        return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    def _process_italic(self, text: str) -> str:
        """处理斜体 *text*"""
        return re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    def _process_link(self, text: str) -> str:
        """处理链接 [text](url)"""
        return re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" style="color: #3498db; text-decoration: none;">\1</a>', text)
    
    def _process_inline_formatting(self, text: str) -> str:
        """处理行内格式化"""
        text = self._process_bold(text)
        text = self._process_italic(text)
        text = self._process_link(text)
        return text
    
    def _format_code_block(self, code_lines: list, lang: str) -> str:
        """格式化代码块"""
        code_text = '\n'.join(code_lines)
        # 简单的 HTML 转义
        code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        lang_label = f'[{lang}]' if lang else ''
        
        return f'''<section style="background: #282c34; padding: 15px; border-radius: 6px; margin: 15px 0; overflow-x: auto;">
<div style="color: #abb2bf; font-family: Consolas, Monaco, monospace; font-size: 13px; line-height: 1.6; white-space: pre-wrap;">{code_text}</div>
</section>'''
    
    def add_wechat_header(self, html_content: str, title: str = "") -> str:
        """添加公众号头部样式"""
        header = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.8;
    color: #333;
    max-width: 677px;
    margin: 0 auto;
    padding: 20px;
}}
</style>
</head>
<body>
'''
        
        if title:
            header += f'<h1 style="font-size: 22px; color: #2c3e50; text-align: center; margin-bottom: 30px;">{title}</h1>'
        
        footer = '''
</body>
</html>'''
        
        return header + html_content + footer


def convert_file(input_path: str, output_path: str = None):
    """
    转换单个文件
    
    Args:
        input_path: 输入 Markdown 文件路径
        output_path: 输出 HTML 文件路径（可选）
    """
    converter = MarkdownToWechat()
    
    # 读取 Markdown 文件
    with open(input_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # 提取标题（第一行的 # 标题）
    title = ""
    first_line = markdown_content.split('\n')[0]
    if first_line.startswith('# '):
        title = first_line[2:]
    
    # 转换
    html_content = converter.convert(markdown_content)
    full_html = converter.add_wechat_header(html_content, title)
    
    # 保存
    if output_path is None:
        output_path = input_path.replace('.md', '_wechat.html')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✅ 转换完成！")
    print(f"📄 输入文件: {input_path}")
    print(f"📄 输出文件: {output_path}")
    print(f"\n💡 使用方法：")
    print(f"   1. 用浏览器打开 {output_path}")
    print(f"   2. 全选复制 (Ctrl+A, Ctrl+C)")
    print(f"   3. 粘贴到公众号编辑器")
    
    return output_path


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("❌ 用法: python md_to_wechat.py <markdown文件路径>")
        print("\n示例:")
        print("  python md_to_wechat.py 公众号首篇.md")
        print("  python md_to_wechat.py article.md output.html")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"❌ 文件不存在: {input_file}")
        sys.exit(1)
    
    convert_file(input_file, output_file)


if __name__ == "__main__":
    main()
