const assert=require('node:assert/strict');
const {test}=require('node:test');
const {commands,copy}=require('../web/console/inference_help.js');
const {spawnSync}=require('node:child_process');
const fs=require('node:fs');

test('two-window templates use saved local port and consistent remote placeholders',()=>{
  const value=commands({host:'127.0.0.1',port:9911});
  assert.match(value.tunnel,/-N -L '127.0.0.1:9911:127.0.0.1:<policy-port>'/);
  assert.match(value.tunnel,/ExitOnForwardFailure=yes/);
  assert.match(value.tunnel,/ServerAliveInterval=15/);
  assert.match(value.tunnel,/ServerAliveCountMax=3/);
  assert.ok(!value.tunnel.includes('QUIET'));
  assert.ok(!value.login.includes('-N'));
  assert.match(value.server,/cd -- "<openpi-project-directory>" \|\| exit 1/);
  assert.match(value.server,/--port "<policy-port>"/);
  assert.match(value.server,/export PATH="\$HOME\/\.local\/bin:\$PATH"/);
  assert.match(value.server,/--policy.dir="<checkpoint-directory>"/);
  assert.ok(!value.server.includes('\n+'));
  assert.match(commands({host:"ws://[::1]/policy",port:8000}).local,/-ComputerName '\[?::1\]?'/);
});

test('PowerShell templates parse without executing ssh', {skip:process.platform!=='win32'},()=>{
  const value=commands({host:'127.0.0.1',port:9911});
  const script=`$text=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${Buffer.from(value.tunnel+'\n'+value.login+'\n'+value.local).toString('base64')}')); $tokens=$null; $parseErrors=$null; [void][Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$parseErrors); if ($parseErrors.Count) { $parseErrors | Out-String; exit 1 }`;
  const result=spawnSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',script],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr+result.stdout);
});

const bash=process.platform==='win32'?'C:/Program Files/Git/bin/bash.exe':'/bin/bash';
test('remote launch is valid Bash and retains literal remote environment variables',{skip:!fs.existsSync(bash)},()=>{
  const script=commands({host:'127.0.0.1',port:8001}).server;
  const result=spawnSync(bash,['-n'],{input:script,encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
});

test('clipboard success preserves exactly the command and failure reports false',async()=>{
  const text=commands({host:'localhost',port:9911}).server;
  let captured,removed=false,focused=false;
  assert.equal(await copy(text,{}, {clipboard:{writeText:async value=>{captured=value;}}}),true);
  assert.equal(captured,text);
  const document={activeElement:{focus(){focused=true;}},body:{append(){}},
    createElement(){return {style:{},select(){},remove(){removed=true;}};},execCommand(){return false;}};
  assert.equal(await copy(text,document,{clipboard:{writeText:async()=>{throw Error('denied');}}}),false);
  assert.ok(removed&&focused);
  document.execCommand=()=>true;
  assert.equal(await copy(text,document,{}),true);
});
