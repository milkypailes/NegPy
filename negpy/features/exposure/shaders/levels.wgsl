// Display-referred levels, the last pipeline step; mirrors levels.py::apply_levels.
//
// The buffer is already display-encoded (output_encode ran). The Global master
// applies equally to all three channels first, then each per-channel curve.
// Uniforms carry normalized bounds over (value, red, green, blue); gamma rides
// inverted, as the pow exponent.

struct LevelsUniforms {
    in_lo: vec4<f32>,
    in_hi: vec4<f32>,
    gamma_inv: vec4<f32>,
    out_lo: vec4<f32>,
    out_span: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LevelsUniforms;

fn level_channel(x: f32, lo: f32, hi: f32, ginv: f32, olo: f32, ospan: f32) -> f32 {
    var t: f32;
    if (hi > lo) {
        t = (x - lo) / (hi - lo);
    } else {
        // Degenerate window: the low marker alone thresholds, as in GIMP, where
        // the unnormalized distance is clamped straight onto [0, 1].
        t = (x - lo) * 255.0;
    }
    t = clamp(t, 0.0, 1.0);
    if (ginv != 1.0) {
        t = pow(t, ginv);
    }
    return olo + ospan * t;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;
    color = vec3<f32>(
        level_channel(color.r, params.in_lo[0], params.in_hi[0], params.gamma_inv[0], params.out_lo[0], params.out_span[0]),
        level_channel(color.g, params.in_lo[0], params.in_hi[0], params.gamma_inv[0], params.out_lo[0], params.out_span[0]),
        level_channel(color.b, params.in_lo[0], params.in_hi[0], params.gamma_inv[0], params.out_lo[0], params.out_span[0])
    );
    color.r = level_channel(color.r, params.in_lo[1], params.in_hi[1], params.gamma_inv[1], params.out_lo[1], params.out_span[1]);
    color.g = level_channel(color.g, params.in_lo[2], params.in_hi[2], params.gamma_inv[2], params.out_lo[2], params.out_span[2]);
    color.b = level_channel(color.b, params.in_lo[3], params.in_hi[3], params.gamma_inv[3], params.out_lo[3], params.out_span[3]);
    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
