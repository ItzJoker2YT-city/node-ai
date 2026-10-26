const esbuild=require('esbuild');const fs=require('fs');
esbuild.buildSync({entryPoints:['frontend/dashboard.jsx'],bundle:true,minify:true,outfile:'dist/dashboard.js',loader:{'.css':'css'}});
const js=fs.readFileSync('dist/dashboard.js','utf8').replace(/<\/script/gi,'<\\/script');
const css=fs.readFileSync('dist/dashboard.css','utf8');
fs.writeFileSync('dist/dashboard.html',`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Node · Private conversation lab</title><style>${css}</style></head><body><div id="root"></div><script>${js}</script></body></html>`);
