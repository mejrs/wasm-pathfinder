/* tslint:disable */
/* eslint-disable */
/**
* WASM interface to Pathfinding::race
* @param {number} plane_1
* @param {number} x_1
* @param {number} y_1
* @param {number} feature_1
* @param {number} plane_2
* @param {number} x_2
* @param {number} y_2
* @param {number} feature_2
* @returns {Promise<any>}
*/
export function race(plane_1: number, x_1: number, y_1: number, feature_1: number, plane_2: number, x_2: number, y_2: number, feature_2: number): Promise<any>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly race: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h3f7a256176343f17: (a: number, b: number, c: number) => void;
  readonly __wbindgen_free: (a: number, b: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly wasm_bindgen__convert__closures__invoke2_mut__h67b371694436836f: (a: number, b: number, c: number, d: number) => void;
}

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
