# 上传 GitHub

本ZIP解压后的 `catalysis-source-adaptation/` 就是仓库根目录。上传的是目录内的代码和文件，不是把整份ZIP当成唯一的仓库内容。

## 推荐：GitHub Desktop

1. 安装GitHub Desktop并登录你的账号。
2. 选择 File → Add local repository，选中解压后的目录；按提示创建本地Git仓库。
3. 提交文件，再选 Publish repository，仓库名可用 `catalysis-source-adaptation`。
4. 若准备公开，取消 Keep this code private，再确认发布。

## 终端方式

先在GitHub创建一个空仓库。以下用户名是你要替换的示例，不是已发布地址。

```bash
cd catalysis-source-adaptation
git init
git add .
git commit -m "Release archived catalysis evaluation code and results"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/catalysis-source-adaptation.git
git push -u origin main
```

本包最大单文件约54MB，低于Git常规仓库的100MiB阻挡线，但超过网页单文件25MiB上限；适合使用Git或GitHub Desktop完整上传。不要因为网页上传失败而删掉预测证据。更大的后续二进制成果可考虑Releases或Git LFS。

官方说明：https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

## 发布后补充

- 在仓库首页填好Description、研究关键词，以及你实际选择的公开仓库地址。
- 如有论文预印本/正式DOI，再更新README和CITATION.cff；当前包不编造DOI、收录或接收状态。
- 保留根目录LICENSE及第三方许可说明；代码的MIT许可不覆盖第三方数据和论文。
- `.gitattributes`保护科研档案的原始字节，避免Windows换行转换破坏校验。

本次只是生成本地开源发布包，没有替你在GitHub创建仓库或发布内容。
