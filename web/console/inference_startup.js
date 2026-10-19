/* Inference-only orchestration; uses the existing session/stop APIs unchanged. */
(function(root) {
  class Startup {
    constructor(options = {}) {
      this.busy = false; this.revision = 0; this.message = '';
      this.now = options.now || (() => Date.now());
      this.sleep = options.sleep || (ms => new Promise(resolve => setTimeout(resolve, ms)));
      this.postTimeoutMs = options.postTimeoutMs || 60000;
      this.totalBudgetMs = options.totalBudgetMs || 90000;
    }
    cancel() { this.revision++; this.message = '正在取消启动，等待硬件准备请求结束；不会继续启用推理'; }
    async run(o) {
      if (this.busy) return false;
      this.busy = true;
      const token = ++this.revision, context = o.context(), began = this.now();
      const valid = () => token === this.revision && context === o.context();
      const check = () => { if (!valid()) { const e = new Error('启动已取消'); e.cancelled = true; throw e; } };
      const notify = text => { this.message = text; o.progress(text); };
      let created = null, timer, outputSession = null;
      const compatible = s => s?.state === 'ACTIVE' && s.client_id === o.clientId && s.execution_mode === o.mode;
      async function cleanup() {
        const owned = created || outputSession;
        if (!owned) return;
        const current = await o.api('/api/osc/state', 'GET', undefined, 3000);
        if (current?.session?.id !== owned.id || current.session.client_id !== o.clientId) return;
        await o.api('/api/pi05/stop', 'POST', {reason:'inference startup cancelled/failed'}, 10000);
        if (created) await o.api('/api/osc/session/stop', 'POST', {reason:'inference startup cancelled/failed'}, 60000);
      }
      o.changed();
      try {
        let current = o.session();
        if (!compatible(current)) {
          const previousId = current?.id;
          const preparation = o.mode === 'hardware' ? '真机会话' : '影子会话';
          notify(`正在准备${preparation}；尚未启用模型运动`);
          timer = setInterval(() => o.progress(this.message = valid()
            ? `正在准备${preparation}，已等待${Math.floor((this.now()-began)/1000)}秒；尚未启用模型运动`
            : '正在取消启动，等待硬件请求返回；不会继续启用推理'), 1000);
          let response;
          try {
            response = await o.api('/api/osc/session/start', 'POST',
              {execution_mode:o.mode,client_id:o.clientId,input_source:'pi05'}, this.postTimeoutMs);
          } catch (error) {
            if (!['AbortError','TimeoutError','TypeError'].includes(error.name)) throw error;
            // The timed-out POST may still be preparing hardware. Never send
            // another start POST; recover only the matching completed session.
            while (this.now()-began < this.totalBudgetMs) {
              let observed;
              try { observed = await o.api('/api/osc/state','GET',undefined,2000); } catch (_) {}
              const s = observed?.session;
              if (compatible(s) && s.id !== previousId) { response = {state:observed}; break; }
              if (s?.state === 'ACTIVE' && s.client_id !== o.clientId) throw new Error('会话属于其他客户端，未启用推理');
              await this.sleep(500);
            }
            if (!response) throw new Error('硬件启动未确认，未启用推理；请检查会话状态，勿重复点击启动');
          }
          current = response?.state?.session;
          if (compatible(current) && current.id !== previousId) created = current;
          if (!compatible(current)) throw new Error('硬件会话尚未就绪，未启用推理');
          o.acceptOsc(response.state);
        }
        clearInterval(timer); timer = null;
        check(); notify('会话已就绪，正在准备推理控制');
        if (o.config) await o.api('/api/pi05/config','POST',o.config,10000);
        check();
        outputSession = current;
        const result = await o.api('/api/pi05/start','POST',{session_id:current.id,client_id:o.clientId},15000);
        check();
        if (result?.state === 'ERROR') throw new Error(result.last_error || '推理准备失败，未保持输出');
        o.acceptPi(result); return true;
      } catch (error) {
        try { await cleanup(); } catch (failure) { error.message += `；会话清理未确认：${failure.message}`; }
        throw error;
      } finally {
        clearInterval(timer); this.busy = false; this.message = ''; o.changed();
      }
    }
  }
  const api = {create:options => new Startup(options)};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.NeroInferenceStartup = api;
})(typeof window !== 'undefined' ? window : globalThis);
