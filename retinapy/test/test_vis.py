import pytest
import retinapy.mea as mea
import retinapy.vis as vis


def test_KernelPlots(four_dc_recs, tmp_path):
    """Tests KernelPlots class.

    Tests that:
        1. Kernel plots generate() function can be called without error.
        2. Loading a kernel plot img after calling generate() works.
    """
    # Setup
    out_dir = tmp_path / "test_KernelPlots"
    out_dir.mkdir()
    # Test
    # 1. Generate plots.
    kernel_plots = vis.KernelPlots.generate(
            four_dc_recs,
            snippet_len=500,
            snippet_pad=50,
            out_dir=out_dir,
            style="mini",
            num_workers=100)
    # 2. Load all expected plots.
    for rec in four_dc_recs:
        for cluster_id in rec.cluster_ids:
            plot_img = kernel_plots.get(rec.name, cluster_id)
            assert plot_img is not None


    



