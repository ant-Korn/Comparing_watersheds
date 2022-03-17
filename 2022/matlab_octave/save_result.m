function save_result(watershed_res, result_save_path)
    mkdir(result_save_path);
    for i = 1:size(watershed_res, 1)       
       imwrite_tiff_uint(squeeze(watershed_res(i,:,:)), fullfile(result_save_path, sprintf("slice_%04d.tif", i)));
    end
end