const esbuild = require('esbuild');
const fs = require('fs');
esbuild.buildSync({entryPoints:['frontend/app.jsx'],bundle:true,minify:true,write:true,outfile:'dist/app.js',loader:{'.css':'css'}});
const js=fs.readFileSync('dist/app.js','utf8').replace(/<\/script/gi,'<\\/script');
const css=fs.readFileSync('dist/app.css','utf8');
for(const preview of [false,true]){
 const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Node · Local coding lab</title><style>${css}</style></head><body><div id="root"></div><script>window.NODE_PREVIEW=${preview};</script><script>${js}</script></body></html>`;
 fs.writeFileSync(preview?'../node-preview.html':'dist/index.html',html);
}
