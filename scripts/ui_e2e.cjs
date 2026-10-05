/* Optional browser QA. npm install playwright; npx playwright install chromium.
   PLAYWRIGHT_MODULE / CHROMIUM_MODULE are optional executor overrides. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium: playwright } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const launchOptions = { headless: true };
  if (process.env.PLAYWRIGHT_CHANNEL) launchOptions.channel = process.env.PLAYWRIGHT_CHANNEL;
  if (process.env.CHROMIUM_MODULE) {
    const module = require(process.env.CHROMIUM_MODULE);
    const binary = module.default || module;
    launchOptions.executablePath = await binary.executablePath();
    launchOptions.args = binary.args;
  }
  const browser = await playwright.launch(launchOptions);
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, locale: 'zh-CN' });
  const page = await context.newPage();
  const errors = [];
  const output = process.env.UI_ARTIFACT_DIR || 'artifacts';
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(e.message));
  try {
    await page.goto(process.env.UI_BASE_URL || 'http://127.0.0.1:5173');
    await page.getByPlaceholder('至少 3 位字母、数字或下划线').fill(`ui_${Date.now()}`);
    await page.getByPlaceholder('至少 8 位').fill('ui-test-password-123');
    await page.getByRole('button', { name: '创建账号', exact: true }).click();
    await page.getByRole('button', { name: '＋ 创建知识库', exact: true }).click();
    await page.getByPlaceholder('知识库名称', { exact: true }).fill('图文复现 · 验收知识库');
    await page.getByRole('button', { name: '创建', exact: true }).click();
    const nav = name => page.locator('nav button').filter({ hasText: name }).click();
    await nav('文档与片段');
    await page.getByRole('button', { name: '载入示例', exact: true }).click();
    await page.getByText('ready', { exact: true }).waitFor({ timeout: 15000 });
    fs.mkdirSync(output, { recursive: true });
    await page.screenshot({ path: path.join(output, 'ui_documents.png'), fullPage: true });
    await nav('知识问答');
    await page.getByRole('button', { name: '出差报销申请需要几天内提交？ ↗', exact: true }).click();
    await page.getByRole('button', { name: '发送 ↗', exact: true }).click();
    await page.locator('.message.assistant .answer-text').filter({ hasText: '7天' }).waitFor({ timeout: 15000 });
    await page.getByRole('button', { name: '发送 ↗', exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'ui_chat.png'), fullPage: true });
    // Real browser refresh retains tab login and accepted history.
    await page.reload();
    await page.locator('.history-item').first().waitFor();
    await page.locator('.history-item').first().click();
    await page.locator('.message.assistant .answer-text').filter({ hasText: '7天' }).waitFor();
    await page.getByPlaceholder('输入问题，Ctrl / ⌘ + Enter 发送').fill('报销申请需要几天内提交？');
    await page.getByRole('button', { name: '发送 ↗', exact: true }).click();
    await page.locator('.draft-box').waitFor();
    await page.getByRole('button', { name: '停止生成', exact: true }).click();
    await page.locator('.draft-box').waitFor({ state: 'hidden' });
    await page.getByRole('button', { name: '发送 ↗', exact: true }).waitFor();
    assert.ok((await page.locator('.message.assistant').last().textContent()).includes('已停止'));
    // Disconnect via page reload while the independent worker keeps running.
    await page.getByPlaceholder('输入问题，Ctrl / ⌘ + Enter 发送').fill('报销申请需要几天内提交？');
    await page.getByRole('button', { name: '发送 ↗', exact: true }).click();
    await page.locator('.draft-box').waitFor();
    await page.reload();
    await page.getByRole('button', { name: '停止生成', exact: true }).waitFor({ timeout: 15000 });
    await page.getByRole('button', { name: '停止生成', exact: true }).waitFor({ state: 'hidden', timeout: 15000 });
    await page.getByRole('button', { name: '发送 ↗', exact: true }).waitFor();
    assert.ok((await page.locator('.message.assistant .answer-text').last().textContent()).includes('7天'));
    assert.equal(await page.locator('.step .running').count(), 0);
    assert.equal(await page.locator('.draft-box').count(), 0);
    await page.screenshot({ path: path.join(output, 'ui_recovery.png'), fullPage: true });
    await nav('检索调试');
    await page.getByPlaceholder('输入查询，查看召回结果').fill('报销申请');
    await page.getByRole('button', { name: '运行检索', exact: true }).click();
    await page.getByRole('heading', { name: '最终证据上下文', exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'ui_retrieval.png'), fullPage: true });
    await nav('检索策略');
    assert.equal(await page.getByRole('switch').count(), 7);
    await nav('质量评测');
    await page.getByRole('button', { name: '运行评测', exact: true }).click();
    await page.getByRole('heading', { name: '质量指标', exact: true }).waitFor({ timeout: 20000 });
    await page.locator('.metric-chart canvas').waitFor();
    await page.screenshot({ path: path.join(output, 'ui_evaluation.png'), fullPage: true });
    // Explicitly mocked UI failure payload; backend Judge failure has separate pytest coverage.
    await page.route('**/api/evaluations', async route => {
      if (route.request().method() !== 'POST') return route.continue();
      await route.fulfill({ json: { data: { status: 'partial_missing', metrics: {
        context_recall: null, context_precision: null, faithfulness: null, answer_relevancy: null },
        results: [{ question: 'Judge 故障样本', status: 'missing', metrics: { context_recall: null, faithfulness: null } }] } } });
    });
    await page.getByRole('button', { name: '运行评测', exact: true }).click();
    await page.getByText('部分评测或 Judge 失败，缺失指标不计入均值。').waitFor();
    await page.getByText('Judge 故障样本', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'ui_missing_judge.png'), fullPage: true });
    await page.unroute('**/api/evaluations');

    await page.getByRole('button', { name: '三档策略对比', exact: true }).click();
    await page.getByRole('heading', { name: '三档策略对比', exact: true }).waitFor({ timeout: 20000 });
    await page.screenshot({ path: path.join(output, 'ui_comparison.png'), fullPage: true });
    await page.getByRole('button', { name: '从文档生成 5 条', exact: true }).click();
    await page.getByRole('button').filter({ hasText: 'document_generated' }).waitFor({ timeout: 20000 });
    await nav('应用管理');
    await page.getByRole('button', { name: '创建应用', exact: true }).click();
    await page.getByPlaceholder('应用名称', { exact: true }).fill('制度客服');
    await page.getByRole('button', { name: '保存应用', exact: true }).click();
    await page.getByText('制度客服', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'ui_applications.png'), fullPage: true });
    await nav('知识问答');
    await page.locator('.chat-panel .el-select').first().click();
    await page.getByRole('option', { name: '制度客服', exact: true }).click();
    await page.getByRole('button', { name: '出差报销申请需要几天内提交？ ↗', exact: true }).click();
    await page.getByRole('button', { name: '发送 ↗', exact: true }).click();
    await page.getByRole('button', { name: '有帮助', exact: true }).waitFor({ timeout: 15000 });
    await page.getByRole('button', { name: '没有帮助', exact: true }).click();
    await nav('对话反馈');
    await page.getByRole('button', { name: '订正为测评样本', exact: true }).click();
    await page.getByPlaceholder('正确的标准答案', { exact: true }).fill('7天内提交报销申请');
    await page.getByPlaceholder('原子事实 JSON 数组，例如 ["保修期为24个月"]', { exact: true }).fill('["7天内提交报销申请"]');
    await page.getByRole('button', { name: '保存测评样本', exact: true }).click();
    await page.getByText('人工标准答案已保存为测评样本', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'ui_feedback.png'), fullPage: true });
    await nav('模型配置');
    await page.getByRole('button', { name: '测试三类模型', exact: true }).click();
    await page.getByText(/embedding_dimensions/).waitFor({ timeout: 15000 });
    await nav('文档与片段');
    const readyRevision = page.waitForResponse(async response => {
      if (!response.url().endsWith('/documents') || response.status() !== 200) return false;
      const body = await response.json();
      return body.data.some(d => d.status === 'ready' && d.index_revision === 2);
    }, { timeout: 20000 });
    const [chooser] = await Promise.all([page.waitForEvent('filechooser'),
      page.getByRole('button', { name: '覆盖文件', exact: true }).click()]);
    await chooser.setFiles({ name: '新版制度.md', mimeType: 'text/markdown',
      buffer: Buffer.from('公司差旅政策规定，员工出差后必须在14天内提交报销申请。') });
    await readyRevision;
    await page.getByRole('button', { name: '片段', exact: true }).click();
    await page.getByText('公司差旅政策规定，员工出差后必须在14天内提交报销申请。', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'ui_replacement.png'), fullPage: true });
    assert.deepEqual(errors, []);
    const result = { passed: true, checks: ['registration', 'knowledge_base', 'sample_upload',
      'ready_status', 'chat_final_answer_and_citations', 'retrieval_debug', 'seven_switches', 'evaluation_chart',
      'three_strategy_comparison', 'document_generated_dataset', 'application_creation_and_chat',
      'feedback_and_manual_dataset', 'model_connectivity_test', 'replacement_file_and_new_revision',
      'refresh_login_and_history', 'explicit_cancel_and_draft_cleanup', 'refresh_active_run_recovery', 'mocked_missing_judge_values_render'],
      page_errors: errors, screenshots: ['ui_documents.png', 'ui_chat.png', 'ui_retrieval.png', 'ui_evaluation.png',
        'ui_comparison.png', 'ui_applications.png', 'ui_feedback.png', 'ui_replacement.png', 'ui_recovery.png', 'ui_missing_judge.png'], mocked_checks: ['mocked_missing_judge_values_render'] };
    fs.writeFileSync(path.resolve(path.join(output, 'ui_e2e.json')), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await context.close(); await browser.close();
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
