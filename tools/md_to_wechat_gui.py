"""
Markdown 转公众号格式工具 - GUI 版本
使用 Tkinter 创建图形界面，无需命令行
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
from pathlib import Path

# 导入转换引擎
sys.path.insert(0, str(Path(__file__).parent))
from md_to_wechat_enhanced import MarkdownToWechatEnhanced, THEMES


class WechatConverterGUI:
    """微信公众号转换器 GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Markdown 转公众号格式工具")
        self.root.geometry("900x700")
        
        # 变量
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.selected_theme = tk.StringVar(value="default")
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="📝 Markdown 转公众号格式工具", 
                               font=("Microsoft YaHei", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 输入文件选择
        ttk.Label(main_frame, text="输入文件：", font=("Microsoft YaHei", 10)).grid(
            row=1, column=0, sticky=tk.W, pady=5)
        
        input_entry = ttk.Entry(main_frame, textvariable=self.input_file, width=60)
        input_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(main_frame, text="浏览...", command=self._browse_input).grid(
            row=1, column=2, pady=5)
        
        # 输出文件选择
        ttk.Label(main_frame, text="输出文件：", font=("Microsoft YaHei", 10)).grid(
            row=2, column=0, sticky=tk.W, pady=5)
        
        output_entry = ttk.Entry(main_frame, textvariable=self.output_file, width=60)
        output_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Button(main_frame, text="浏览...", command=self._browse_output).grid(
            row=2, column=2, pady=5)
        
        # 主题选择
        theme_frame = ttk.LabelFrame(main_frame, text="🎨 主题选择", padding="10")
        theme_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        themes_row = ttk.Frame(theme_frame)
        themes_row.pack(fill=tk.X)
        
        for idx, (theme_id, config) in enumerate(THEMES.items()):
            rb = ttk.Radiobutton(themes_row, text=f"{config['name']} ({theme_id})", 
                                variable=self.selected_theme, value=theme_id)
            rb.grid(row=0, column=idx, padx=10, sticky=tk.W)
        
        # 预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="👁️ 预览", padding="10")
        preview_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=15, width=80,
                                                      wrap=tk.WORD, font=("Consolas", 10))
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="🔄 预览转换", command=self._preview_conversion,
                  width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="✅ 开始转换", command=self._start_conversion,
                  width=15, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="📂 打开输出文件夹", command=self._open_output_folder,
                  width=18).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="❌ 清空", command=self._clear,
                  width=10).pack(side=tk.LEFT, padx=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 配置网格权重
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
    
    def _browse_input(self):
        """浏览输入文件"""
        filename = filedialog.askopenfilename(
            title="选择 Markdown 文件",
            filetypes=[("Markdown 文件", "*.md"), ("所有文件", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            # 自动设置输出文件名
            if not self.output_file.get():
                output = filename.replace('.md', '_wechat.html')
                self.output_file.set(output)
    
    def _browse_output(self):
        """浏览输出文件"""
        filename = filedialog.asksaveasfilename(
            title="保存 HTML 文件",
            defaultextension=".html",
            filetypes=[("HTML 文件", "*.html"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
    
    def _preview_conversion(self):
        """预览转换结果"""
        input_path = self.input_file.get()
        if not input_path or not Path(input_path).exists():
            messagebox.showerror("错误", "请先选择有效的输入文件！")
            return
        
        try:
            self.status_var.set("正在预览...")
            self.root.update()
            
            # 读取文件
            with open(input_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # 转换
            converter = MarkdownToWechatEnhanced(theme=self.selected_theme.get())
            html_content = converter.convert(markdown_content)
            
            # 显示预览（前 2000 字符）
            preview = html_content[:2000] + "\n\n...（更多内容将在转换后查看）"
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, preview)
            
            self.status_var.set(f"预览完成 | 主题：{THEMES[self.selected_theme.get()]['name']}")
            
        except Exception as e:
            messagebox.showerror("错误", f"预览失败：\n{str(e)}")
            self.status_var.set("预览失败")
    
    def _start_conversion(self):
        """开始转换"""
        input_path = self.input_file.get()
        output_path = self.output_file.get()
        
        if not input_path or not Path(input_path).exists():
            messagebox.showerror("错误", "请先选择有效的输入文件！")
            return
        
        if not output_path:
            messagebox.showerror("错误", "请指定输出文件路径！")
            return
        
        try:
            self.status_var.set("正在转换...")
            self.root.update()
            
            # 读取文件
            with open(input_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # 提取标题
            title = ""
            first_line = markdown_content.split('\n')[0]
            if first_line.startswith('# '):
                title = first_line[2:]
            
            # 转换
            converter = MarkdownToWechatEnhanced(theme=self.selected_theme.get())
            html_content = converter.convert(markdown_content)
            full_html = converter.add_wechat_header(html_content, title)
            
            # 保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_html)
            
            self.status_var.set(f"✅ 转换成功！| {Path(output_path).name}")
            
            # 显示完整预览
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, full_html)
            
            messagebox.showinfo("成功", 
                              f"转换完成！\n\n"
                              f"输出文件：{output_path}\n"
                              f"使用主题：{THEMES[self.selected_theme.get()]['name']}\n\n"
                              f"下一步：\n"
                              f"1. 用浏览器打开 HTML 文件\n"
                              f"2. 全选复制 (Ctrl+A, Ctrl+C)\n"
                              f"3. 粘贴到公众号编辑器")
            
        except Exception as e:
            messagebox.showerror("错误", f"转换失败：\n{str(e)}")
            self.status_var.set("转换失败")
    
    def _open_output_folder(self):
        """打开输出文件夹"""
        output_path = self.output_file.get()
        if output_path:
            folder = str(Path(output_path).parent)
            import subprocess
            subprocess.Popen(f'explorer "{folder}"')
        else:
            messagebox.showwarning("提示", "请先指定输出文件路径")
    
    def _clear(self):
        """清空所有内容"""
        self.input_file.set("")
        self.output_file.set("")
        self.preview_text.delete(1.0, tk.END)
        self.status_var.set("就绪")


def main():
    """主函数"""
    root = tk.Tk()
    app = WechatConverterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
