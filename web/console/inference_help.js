/* Deployment help contains templates only; no credentials or remote execution. */
(function(root) {
  "use strict";
  function commands(preset, execution={}) {
    let host=preset.host;
    if (/^wss?:\/\//.test(host)) {try {host=new URL(host).hostname;}catch {}}
    return {
      tunnel:`ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -N -L '127.0.0.1:${preset.port}:127.0.0.1:<policy-port>' -p '<ssh-port>' '<user>@<remote-host>'`,
      login:"ssh -p '<ssh-port>' '<user>@<remote-host>'",
      server:(execution.mode==="rtc"?[
        'cd -- "<openpi-project-directory>" || exit 1',
        'export PATH="$HOME/.local/bin:$PATH"',
        'export BOTO_CONFIG="$HOME/.boto"',
        'export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"',
        'exec uv run --frozen --no-sync python "<rtc-service-directory>/serve.py" \\',
        '  --checkpoint "<checkpoint-directory>" \\',
        '  --port "<policy-port>" --validate'
      ]:[
        'cd -- "<openpi-project-directory>" || exit 1',
        'export PATH="$HOME/.local/bin:$PATH"',
        'export BOTO_CONFIG="$HOME/.boto"',
        'exec uv run --frozen --no-sync scripts/serve_policy.py \\',
        '  --port "<policy-port>" \\',
        '  policy:checkpoint \\',
        '  --policy.config="<training-config-name>" \\',
        '  --policy.dir="<checkpoint-directory>"'
      ]).join("\n"),
      local:`Test-NetConnection -ComputerName '${host.replace(/'/g,"''")}' -Port ${preset.port}`
    };
  }
  function markup() {
    return `<details class="pi05-help"><summary>启动与连接指南</summary><div class="help-content">
      <p id="inference-help"></p><p id="inference-help-draft" class="help-note"></p>
      <div id="inference-autodl-guide">
        <p>将所有 &lt;占位符&gt; 替换为实际部署值。两个窗口使用相同的 SSH 用户、主机与 SSH 端口。</p>
        <div class="inference-help-cards">
          <section class="inference-help-card"><strong>窗口 A · 建立并保持转发</strong><small>本机 Windows PowerShell</small>
            <pre><code id="inference-help-command"></code></pre><button type="button" class="button quiet" data-copy-command="inference-help-command">复制转发命令</button>
            <p>保持窗口打开。登录后没有持续输出通常是正常现象；以服务入口和 WebSocket 状态确认连接。模型已运行时，只需此窗口。</p>
          </section>
          <section class="inference-help-card"><strong>窗口 B · 登录并启动模型</strong><small>① 本机 Windows PowerShell：登录 AutoDL</small>
            <pre><code id="inference-help-login"></code></pre><button type="button" class="button quiet" data-copy-command="inference-help-login">复制登录命令</button>
            <small>② 登录后，粘贴到远端 AutoDL Bash</small><pre><code id="inference-help-server"></code></pre><button type="button" class="button quiet" data-copy-command="inference-help-server">复制模型启动命令</button>
            <p>保留窗口查看模型加载与推理日志。模型已经运行时，不要重复启动。</p>
          </section>
        </div>
        <p><code id="inference-help-route"></code></p>
        <p>本机入口端口可以与远端不同；转发命令中的 &lt;policy-port&gt; 必须与模型启动的 --port 一致。SSH 的 -p 是登录端口，与模型端口不同。转发目标中的 127.0.0.1 指 AutoDL 自身。</p>
        <p id="inference-help-dataflow">控制台自动建立 WebSocket，发送双路 RGB 图像、7D 状态与任务文本，接收 H×7 动作并执行前 K 步。无需第三个 WebSocket 终端。H 不决定训练配置或检查点。</p>
        <p>先确认模型 WebSocket 已连接，再等待 7D／H 响应验证通过；输出需要手动开始。关闭窗口 A 会断开隧道；页面停止输出不会关闭 SSH 或远端模型。</p>
        <details><summary>连接失败时如何检查</summary><ul>
          <li>入口未监听：检查窗口 A 的 SSH 登录和转发报错。端口占用时，核实占用程序，或修改预设端口并保存，再复制新命令。</li>
          <li>SSH 登录失败：核对用户名、主机、SSH 端口以及 AutoDL 实例是否运行。</li>
          <li>入口绿色但 WebSocket 失败：入口存在不证明远端正常；检查窗口 B 的模型是否加载完成、远端端口是否一致，并查看握手错误。</li>
          <li>模型启动失败：检查项目目录、uv 环境、训练配置名、检查点与归一化文件是否匹配；以远端日志为准。</li>
          <li>连接成功但数据契约不匹配：检查模型输入／动作是否为 7D、返回步数是否为 H。网页修改 H 不会修改已加载模型。</li>
        </ul></details>
      </div>
      <div id="inference-local-guide"><p>先启动本地或 WSL 的 OpenPI 模型服务，并确保配置端点可从控制台所在主机访问。本地直连无需 SSH。</p><small>本机 Windows PowerShell · 可选手动端口检查</small><pre><code id="inference-help-local-command"></code></pre><button type="button" class="button quiet" data-copy-command="inference-help-local-command">复制检查命令</button><p>服务入口可达后，还需完成 WebSocket 连接及 7D／H 响应验证。H/K 不决定模型启动命令或检查点。</p></div>
      <p id="inference-copy-result" role="status" aria-live="polite"></p>
    </div></details>`;
  }
  async function copy(text, document, navigator) {
    try {
      if (!navigator?.clipboard?.writeText) throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const area=document.createElement("textarea");
      area.value=text;area.style.position="fixed";area.style.opacity="0";
      const previous=document.activeElement;
      document.body.append(area);
      try {area.select();return Boolean(document.execCommand?.("copy"));}
      catch {return false;}
      finally {area.remove();previous?.focus();}
    }
  }
  function bind() {
    document.querySelectorAll("[data-copy-command]").forEach(button => {
      button.onclick=async()=>{
        const ok=await copy(document.getElementById(button.dataset.copyCommand).textContent,document,root.navigator);
        document.getElementById("inference-copy-result").textContent=ok?"命令已复制；请替换占位符后，在标明的终端中执行":"复制失败，请选中命令手动复制";
      };
    });
  }
  function render(name,preset,dirty,execution={}) {
    const get=id=>document.getElementById(id), value=commands(preset,execution), autodl=name==="autodl";
    let host=preset.host;
    if (/^wss?:\/\//.test(host)) {try{host=new URL(host).hostname;}catch {}}
    const loopback=host==="localhost" || host==="::1" || host==="[::1]" || /^127\./.test(host);
    get("inference-autodl-guide").hidden=!autodl;
    get("inference-local-guide").hidden=autodl;
    get("inference-help").textContent=autodl
      ? (loopback?`AutoDL 预设已保存的本机入口：${preset.host}:${preset.port}。按下面两个窗口分别操作。`:`当前为远端直连：${preset.host}:${preset.port}。下面的 SSH 转发模板不适用于当前直连配置；如需转发，请先将预设地址改成本机回环地址并保存。`)
      : `5090 预设已保存的端点：${preset.host}:${preset.port}。`;
    get("inference-help-draft").textContent=dirty?"以下命令使用所选预设的已保存值；存在未应用的设置，请先保存并应用，再复制命令。":"";
    get("inference-help-dataflow").textContent=execution.mode==="rtc"
      ? "RTC：控制台自动发送双 RGB、7D 状态、任务文本和 RTC 前缀，接收20×7动作，以50 ms时间轴执行；K不参与RTC。远端需安装本仓库 inference/rtc 服务文件，默认进行影子验证。无需第三个 WebSocket 终端。"
      : "控制台自动建立 WebSocket，发送双路 RGB 图像、7D 状态与任务文本，接收 H×7 动作并执行前 K 步。无需第三个 WebSocket 终端。H 不决定训练配置或检查点。";
    for(const [id,key] of [["inference-help-command","tunnel"],["inference-help-login","login"],["inference-help-server","server"],["inference-help-local-command","local"]]) get(id).textContent=value[key];
    get("inference-help-route").textContent=`本机控制台 → 127.0.0.1:${preset.port} → SSH 隧道 → AutoDL 127.0.0.1:<policy-port>`;
  }
  const api={commands,markup,bind,render,copy};
  if(typeof module!=="undefined"&&module.exports) module.exports=api;
  else root.NeroInferenceHelp=api;
})(typeof window!=="undefined"?window:globalThis);
