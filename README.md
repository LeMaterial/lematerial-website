# HOW to add content

### install

Download hugo 0.128.0 (wget https://github.com/gohugoio/hugo/releases/download/v0.128.0/hugo_extended_0.128.0_linux-amd64.deb) 
and install npm

```bash
npm install
```


TO start locally:

`hugo server -D --baseURL http://localhost:1313/`


index is at layout/index.html

content (doc) is at content/docs

### deploy

```bash
hugo
npm run deploy
```

