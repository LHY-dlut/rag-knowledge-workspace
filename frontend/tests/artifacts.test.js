import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { graphOption, mergeArtifacts, numericOption } from '../src/artifact_view.js'

test('replay and terminal compensation merge artifacts by ID', () => {
  assert.deepEqual(mergeArtifacts([{id:'1',current_evidence:true}], [{id:'1',current_evidence:false}, {id:'2'}]), [{id:'1',current_evidence:false}, {id:'2'}])
})
test('chart options reject invalid values and never execute payload options', () => {
  const result = numericOption({unit:'名',formatter:'alert(1)',points:[{value:17,label:'<img onerror=alert(1)>'},{value:true},{value:Infinity}]})
  assert.deepEqual(result.series[0].data, [17])
  assert.equal(result.tooltip.renderMode, 'richText')
  assert.equal(result.tooltip.formatter, undefined)
  assert.equal(result.yAxis.name, '名')
  assert.equal(result.yAxis.min, 0)
})
test('negative values retain a zero reference without clipping', () => {
  assert.equal(numericOption({unit:'吨',points:[{value:-2,label:'净流量'}]}).yAxis.min, -2)
})
test('evidence graph rejects edges to nonexistent nodes', () => {
  const result = graphOption({nodes:[{id:'S1',name:'原文',category:'source'},{id:'F1',name:'事实'}],edges:[{source:'S1',target:'F1'},{source:'S999',target:'F1'}]})
  assert.equal(result.series[0].links.length, 1)
  assert.equal(result.tooltip.renderMode, 'richText')
})
test('preview has empty sandbox and no trusted model HTML injection', () => {
  const component = readFileSync(new URL('../src/ArtifactCard.vue', import.meta.url), 'utf8')
  assert.match(component, /sandbox=""/)
  assert.match(component, /referrerpolicy="no-referrer"/)
  assert.doesNotMatch(component, /v-html|allow-scripts|allow-same-origin/)
})
