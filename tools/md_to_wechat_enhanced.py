"""
Markdown 转公众号格式工具 - 增强版
支持：多主题、表格、数学公式、图片处理
"""
import re
import sys
import json
from pathlib import Path
from typing import Dict, Optional


# ==================== 主题配置 ====================

THEMES = {
    "default": {
        "name": "默认主题",
        "colors": {
            "primary": "#2c3e50",      # 主色调（标题）
            "secondary": "#34495e",    # 次色调
            "accent": "#3498db",       # 强调色（链接、引用边框）
            "text": "#333333",         # 正文颜色
            "code_bg": "#282c34",      # 代码块背景
            "code_text": "#abb2bf",    # 代码文字
            "quote_bg": "#f8f9fa",     # 引用块背景
            "border": "#eeeeee",       # 边框颜色
        },
        "fonts": {
            "title_h1": "22px",
            "title_h2": "18px",
            "title_h3": "16px",
            "body": "15px",
            "code": "13px",
            "line_height": "1.8",
        }
    },
    
    "light": {
        "name": "浅色主题",
        "colors": {
            "primary": "#1a1a1a",
            "secondary": "#4a4a4a",
            "accent": "#007aff",
            "text": "#333333",
            "code_bg": "#f5f5f5",
            "code_text": "#333333",
            "quote_bg": "#fafafa",
            "border": "#e0e0e0",
        },
        "fonts": {
            "title_h1": "22px",
            "title_h2": "18px",
            "title_h3": "16px",
            "body": "15px",
            "code": "13px",
            "line_height": "1.8",
        }
    },
    
    "dark": {
        "name": "深色主题",
        "colors": {
            "primary": "#ffffff",
            "secondary": "#e0e0e0",
            "accent": "#64b5f6",
            "text": "#e0e0e0",
            "code_bg": "#1e1e1e",
            "code_text": "#d4d4d4",
            "quote_bg": "#2d2d2d",
            "border": "#404040",
        },
        "fonts": {
            "title_h1": "22px",
            "title_h2": "18px",
            "title_h3": "16px",
            "body": "15px",
            "code": "13px",
            "line_height": "1.8",
        }
    },
    
    "colorful": {
        "name": "彩色主题",
        "colors": {
            "primary": "#ff6b6b",
            "secondary": "#4ecdc4",
            "accent": "#45b7d1",
            "text": "#2d3436",
            "code_bg": "#2d3436",
            "code_text": "#dfe6e9",
            "quote_bg": "#ffeaa7",
            "border": "#fdcb6e",
        },
        "fonts": {
            "title_h1": "22px",
            "title_h2": "18px",
            "title_h3": "16px",
            "body": "15px",
            "code": "13px",
            "line_height": "1.8",
        }
    },
    
    "minimal": {
        "name": "极简主题",
        "colors": {
            "primary": "#000000",
            "secondary": "#333333",
            "accent": "#000000",
            "text": "#000000",
            "code_bg": "#f0f0f0",
            "code_text": "#000000",
            "quote_bg": "#ffffff",
            "border": "#000000",
        },
        "fonts": {
            "title_h1": "24px",
            "title_h2": "20px",
            "title_h3": "17px",
            "body": "16px",
            "code": "14px",
            "line_height": "2.0",
        }
    }
}


class MarkdownToWechatEnhanced:
    """增强版 Markdown 转公众号格式转换器"""
    
    def __init__(self, theme: str = "default"):
        """
        初始化转换器
        
        Args:
            theme: 主题名称 (default/light/dark/colorful/minimal)
        """
        if theme not in THEMES:
            print(f"⚠️  未知主题 '{theme}'，使用默认主题")
            theme = "default"
        
        self.theme = theme
        self.config = THEMES[theme]
        self.colors = self.config["colors"]
        self.fonts = self.config["fonts"]
    
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
        in_table = False
        table_rows = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 处理代码块
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_lang = line.strip()[3:].strip()
                    code_lines = []
                else:
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
            
            # 处理表格
            if self._is_table_separator(line):
                in_table = True
                table_rows = []
                i += 1
                continue
            
            if in_table:
                if line.strip() == '' or line.startswith('|'):
                    if line.startswith('|'):
                        table_rows.append(line)
                    else:
                        # 表格结束
                        table_html = self._format_table(table_rows)
                        result.append(table_html)
                        in_table = False
                        table_rows = []
                        if line.strip() == '':
                            result.append('<br>')
                else:
                    # 非表格行，输出之前的表格
                    if table_rows:
                        table_html = self._format_table(table_rows)
                        result.append(table_html)
                    in_table = False
                    table_rows = []
                    # 继续处理当前行
                    i -= 1
                i += 1
                continue
            
            # 处理数学公式（LaTeX）
            if '$$' in line:
                formula_html = self._format_math_formula(line)
                result.append(formula_html)
                i += 1
                continue
            
            # 处理图片
            if line.strip().startswith('!['):
                img_html = self._format_image(line)
                result.append(img_html)
                i += 1
                continue
            
            # 处理标题
            if line.startswith('# '):
                result.append(self._format_heading(line[2:], 1))
            elif line.startswith('## '):
                result.append(self._format_heading(line[3:], 2))
            elif line.startswith('### '):
                result.append(self._format_heading(line[4:], 3))
            
            # 处理引用块
            elif line.startswith('> '):
                quote_text = line[2:]
                quote_text = self._process_inline_formatting(quote_text)
                result.append(self._format_quote(quote_text))
            
            # 处理列表项
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                item_text = line.strip()[2:]
                item_text = self._process_inline_formatting(item_text)
                result.append(self._format_list_item(item_text))
            
            # 处理有序列表
            elif re.match(r'^\d+\.\s', line.strip()):
                item_text = re.sub(r'^\d+\.\s', '', line.strip())
                item_text = self._process_inline_formatting(item_text)
                result.append(self._format_list_item(item_text))
            
            # 处理分隔线
            elif line.strip() == '---':
                result.append(self._format_divider())
            
            # 处理空行
            elif line.strip() == '':
                result.append('<br>')
            
            # 处理普通段落
            else:
                processed_line = self._process_inline_formatting(line)
                if processed_line.strip():
                    result.append(self._format_paragraph(processed_line))
            
            i += 1
        
        # 处理未闭合的表格
        if in_table and table_rows:
            table_html = self._format_table(table_rows)
            result.append(table_html)
        
        return '\n'.join(result)
    
    def _format_heading(self, text: str, level: int) -> str:
        """格式化标题"""
        text = self._process_inline_formatting(text)
        
        if level == 1:
            size = self.fonts["title_h1"]
            color = self.colors["primary"]
        elif level == 2:
            size = self.fonts["title_h2"]
            color = self.colors["secondary"]
        else:
            size = self.fonts["title_h3"]
            color = self.colors["secondary"]
        
        margin = f"{20 - level * 3}px"
        
        return f'<section style="margin: {margin} 0;"><strong style="font-size: {size}; color: {color};">{text}</strong></section>'
    
    def _format_paragraph(self, text: str) -> str:
        """格式化段落"""
        return f'<p style="margin: 10px 0; line-height: {self.fonts["line_height"]}; font-size: {self.fonts["body"]}; color: {self.colors["text"]};">{text}</p>'
    
    def _format_quote(self, text: str) -> str:
        """格式化引用块"""
        return f'<section style="background: {self.colors["quote_bg"]}; padding: 12px 15px; border-left: 4px solid {self.colors["accent"]}; margin: 10px 0; border-radius: 4px;"><span style="color: #555; font-size: {self.fonts["body"]};">{text}</span></section>'
    
    def _format_list_item(self, text: str) -> str:
        """格式化列表项"""
        return f'<section style="margin: 8px 0; padding-left: 20px;"><span style="color: {self.colors["text"]}; font-size: {self.fonts["body"]};">• {text}</span></section>'
    
    def _format_divider(self) -> str:
        """格式化分隔线"""
        return f'<hr style="border: none; border-top: 1px solid {self.colors["border"]}; margin: 20px 0;">'
    
    def _format_code_block(self, code_lines: list, lang: str) -> str:
        """格式化代码块"""
        code_text = '\n'.join(code_lines)
        code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        lang_label = f'[{lang}]' if lang else ''
        
        return f'''<section style="background: {self.colors["code_bg"]}; padding: 15px; border-radius: 6px; margin: 15px 0; overflow-x: auto;">
<div style="color: {self.colors["code_text"]}; font-family: Consolas, Monaco, monospace; font-size: {self.fonts["code"]}; line-height: 1.6; white-space: pre-wrap;">{code_text}</div>
</section>'''
    
    def _format_table(self, rows: list) -> str:
        """格式化表格"""
        if not rows:
            return ''
        
        # 解析表头和数据行
        headers = []
        data_rows = []
        
        for idx, row in enumerate(rows):
            cells = [cell.strip() for cell in row.split('|')[1:-1]]
            if idx == 0:
                headers = cells
            else:
                data_rows.append(cells)
        
        # 生成 HTML 表格
        html_parts = ['<section style="overflow-x: auto; margin: 15px 0;">']
        html_parts.append('<table style="border-collapse: collapse; width: 100%; font-size: 14px;">')
        
        # 表头
        html_parts.append('<thead><tr>')
        for header in headers:
            html_parts.append(f'<th style="border: 1px solid {self.colors["border"]}; padding: 10px; background: {self.colors["quote_bg"]}; color: {self.colors["primary"]}; font-weight: bold;">{header}</th>')
        html_parts.append('</tr></thead>')
        
        # 数据行
        html_parts.append('<tbody>')
        for row_idx, row in enumerate(data_rows):
            bg_color = '#ffffff' if row_idx % 2 == 0 else self.colors["quote_bg"]
            html_parts.append(f'<tr style="background: {bg_color};">')
            for cell in row:
                cell = self._process_inline_formatting(cell)
                html_parts.append(f'<td style="border: 1px solid {self.colors["border"]}; padding: 10px; color: {self.colors["text"]};">{cell}</td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody>')
        
        html_parts.append('</table>')
        html_parts.append('</section>')
        
        return '\n'.join(html_parts)
    
    def _format_math_formula(self, line: str) -> str:
        """格式化数学公式（简单处理，实际需要使用 MathJax 或 KaTeX）"""
        # 提取公式内容
        formula = line.strip()
        
        # 移除 $$ 标记
        if formula.startswith('$$') and formula.endswith('$$'):
            formula = formula[2:-2].strip()
        elif formula.startswith('$') and formula.endswith('$'):
            formula = formula[1:-1].strip()
        
        # 简单的 LaTeX 到 HTML 转换（仅支持基础符号）
        formula_html = self._latex_to_html(formula)
        
        return f'<section style="background: {self.colors["quote_bg"]}; padding: 15px; margin: 15px 0; border-radius: 4px; text-align: center; font-family: "Times New Roman", serif; font-size: 16px; color: {self.colors["text"]};">{formula_html}</section>'
    
    def _latex_to_html(self, latex: str) -> str:
        """简单的 LaTeX 到 HTML 转换"""
        # 替换常见符号
        replacements = {
            r'\alpha': 'α',
            r'\beta': 'β',
            r'\gamma': 'γ',
            r'\delta': 'δ',
            r'\epsilon': 'ε',
            r'\theta': 'θ',
            r'\lambda': 'λ',
            r'\mu': 'μ',
            r'\pi': 'π',
            r'\sigma': 'σ',
            r'\omega': 'ω',
            r'\sum': '∑',
            r'\int': '∫',
            r'\infty': '∞',
            r'\neq': '≠',
            r'\leq': '≤',
            r'\geq': '≥',
            r'\approx': '≈',
            r'\times': '×',
            r'\div': '÷',
            r'\pm': '±',
            r'\sqrt': '√',
            r'^2': '²',
            r'^3': '³',
        }
        
        for latex_sym, html_sym in replacements.items():
            latex = latex.replace(latex_sym, html_sym)
        
        # 处理上下标
        latex = re.sub(r'\^(\w+)', r'<sup>\1</sup>', latex)
        latex = re.sub(r'_(\w+)', r'<sub>\1</sub>', latex)
        
        return latex
    
    def _format_image(self, line: str) -> str:
        """格式化图片"""
        # 提取图片描述和路径
        match = re.match(r'!\[(.*?)\]\((.*?)\)', line.strip())
        if match:
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # 注意：公众号需要手动上传图片，这里只是占位符
            return f'<section style="text-align: center; margin: 15px 0;"><p style="color: #999; font-size: 13px;">📷 图片：{alt_text}（请在公众号编辑器中手动上传）</p></section>'
        
        return ''
    
    def _is_table_separator(self, line: str) -> bool:
        """判断是否为表格分隔行"""
        return bool(re.match(r'^\|?\s*[-:]+\s*\|', line.strip()))
    
    def _process_bold(self, text: str) -> str:
        """处理加粗"""
        return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    def _process_italic(self, text: str) -> str:
        """处理斜体"""
        return re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    def _process_link(self, text: str) -> str:
        """处理链接"""
        return re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" style="color: {self.colors["accent"]}; text-decoration: none;">\1</a>', text)
    
    def _process_inline_formatting(self, text: str) -> str:
        """处理行内格式化"""
        text = self._process_bold(text)
        text = self._process_italic(text)
        text = self._process_link(text)
        return text
    
    def add_wechat_header(self, html_content: str, title: str = "") -> str:
        """添加公众号头部样式"""
        header = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: {self.fonts["line_height"]};
    color: {self.colors["text"]};
    max-width: 677px;
    margin: 0 auto;
    padding: 20px;
}}
</style>
</head>
<body>
'''
        
        if title:
            header += f'<h1 style="font-size: {self.fonts["title_h1"]}; color: {self.colors["primary"]}; text-align: center; margin-bottom: 30px;">{title}</h1>'
        
        footer = '''
</body>
</html>'''
        
        return header + html_content + footer


def list_themes():
    """列出所有可用主题"""
    print("\n🎨 可用主题：\n")
    for theme_id, config in THEMES.items():
        print(f"  • {theme_id:12s} - {config['name']}")
    print()


def convert_file(input_path: str, output_path: str = None, theme: str = "default"):
    """
    转换单个文件
    
    Args:
        input_path: 输入 Markdown 文件路径
        output_path: 输出 HTML 文件路径（可选）
        theme: 主题名称
    """
    converter = MarkdownToWechatEnhanced(theme=theme)
    
    # 读取 Markdown 文件
    with open(input_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # 提取标题
    title = ""
    first_line = markdown_content.split('\n')[0]
    if first_line.startswith('# '):
        title = first_line[2:]
    
    # 转换
    html_content = converter.convert(markdown_content)
    full_html = converter.add_wechat_header(html_content, title)
    
    # 保存
    if output_path is None:
        theme_suffix = f"_{theme}" if theme != "default" else ""
        output_path = input_path.replace('.md', f'{theme_suffix}_wechat.html')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✅ 转换完成！")
    print(f"📄 输入文件: {input_path}")
    print(f"📄 输出文件: {output_path}")
    print(f"🎨 使用主题: {THEMES[theme]['name']}")
    print(f"\n💡 使用方法：")
    print(f"   1. 用浏览器打开 {output_path}")
    print(f"   2. 全选复制 (Ctrl+A, Ctrl+C)")
    print(f"   3. 粘贴到公众号编辑器")
    
    return output_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Markdown 转公众号格式工具 - 增强版')
    parser.add_argument('input', nargs='?', help='输入 Markdown 文件路径')
    parser.add_argument('-o', '--output', help='输出 HTML 文件路径', default=None)
    parser.add_argument('-t', '--theme', help='主题名称', default='default', 
                       choices=list(THEMES.keys()))
    parser.add_argument('--list-themes', help='列出所有可用主题', action='store_true')
    
    args = parser.parse_args()
    
    if args.list_themes:
        list_themes()
        return
    
    if not args.input:
        print("❌ 请提供输入文件路径")
        print("\n用法: python md_to_wechat_enhanced.py <文件路径> [选项]")
        print("\n示例:")
        print("  python md_to_wechat_enhanced.py article.md")
        print("  python md_to_wechat_enhanced.py article.md -t light")
        print("  python md_to_wechat_enhanced.py --list-themes")
        sys.exit(1)
    
    input_file = args.input
    output_file = args.output
    theme = args.theme
    
    if not Path(input_file).exists():
        print(f"❌ 文件不存在: {input_file}")
        sys.exit(1)
    
    convert_file(input_file, output_file, theme)


if __name__ == "__main__":
    main()
