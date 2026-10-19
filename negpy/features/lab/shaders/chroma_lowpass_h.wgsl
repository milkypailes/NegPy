// Horizontal half of the Chroma Denoise low-pass on CIELAB a*/b*. Taps come from
// gaussian_kernel_1d in features/lab/logic.py, uploaded verbatim (chroma_k buffer), so the
// kernel matches the CPU cv2.sepFilter2D call. Writes (h-blurred a*, h-blurred b*, 0, 0);
// chroma_lowpass_v.wgsl finishes the blur and lab.wgsl reads the result.
struct LabUniforms {
    sharpen: f32,
    chroma_denoise: f32,
    saturation: f32,
    glow_amount: f32,
    halation_strength: f32,
    scale_factor: f32,
    sharpen_radius_px: f32,
    sharpen_masking: f32,
    sharpen_method: f32,
    skin_protection: f32,
    chroma_radius_px: f32,
    _pad3: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LabUniforms;
@group(0) @binding(3) var<storage, read> kernel_w: array<f32>;

// Working-space TRC (Adobe RGB 1998: pure 563/256 gamma) — mirrors oetf_decode in lab.wgsl.
fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return pow(e, vec3<f32>(2.19921875));
}

// CIELAB a*/b* of linear Adobe RGB (D65) — the chroma half of rgb_to_lab in lab.wgsl.
fn lab_ab(rgb: vec3<f32>) -> vec2<f32> {
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);
    var x = (r * 0.5767309 + g * 0.1855540 + b * 0.1881852) / 0.95047;
    var y = r * 0.2973769 + g * 0.6273491 + b * 0.0752741;
    var z = (r * 0.0270343 + g * 0.0706872 + b * 0.9911085) / 1.08883;
    if (x > 0.008856) { x = pow(x, 1.0/3.0); } else { x = (7.787 * x) + (16.0 / 116.0); }
    if (y > 0.008856) { y = pow(y, 1.0/3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    if (z > 0.008856) { z = pow(z, 1.0/3.0); } else { z = (7.787 * z) + (16.0 / 116.0); }
    return vec2<f32>(500.0 * (x - y), 200.0 * (y - z));
}

// cv2's default border mode (BORDER_REFLECT_101) — mirrors the CPU blur.
fn reflect_101(c: i32, n: i32) -> i32 {
    var v = c;
    if (v < 0) { v = -v; }
    if (v >= n) { v = 2 * (n - 1) - v; }
    return clamp(v, 0, n - 1);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));

    let r = i32(params.chroma_radius_px);
    var acc = vec2<f32>(0.0);
    for (var i = -r; i <= r; i++) {
        let sx = reflect_101(coords.x + i, i32(dims.x));
        acc += lab_ab(oetf_decode(textureLoad(input_tex, vec2<i32>(sx, coords.y), 0).rgb)) * kernel_w[u32(i + r)];
    }
    textureStore(output_tex, coords, vec4<f32>(acc, 0.0, 0.0));
}
