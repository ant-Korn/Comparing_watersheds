function img = read_3d_image(folder)
    search_path = fullfile(folder, '*.png');
    fileList = dir(search_path);
    files_num = size(fileList, 1);
    if files_num == 0
        img = 0;
        return;
    end
    first_slice = imread(fullfile(folder, fileList(1,:).name));
    img = zeros([files_num, size(first_slice)], class(first_slice));
    img(1,:,:) = first_slice;
    
    fprintf("\t\tCreated image with size %dx%dx%d\n" , size(img));
    msg = sprintf("\t\tSlices readed: %d/%d\n", 1, files_num);
    fprintf(msg);
    
    for i = 2:files_num
        img(i,:,:) = imread(fullfile(folder, fileList(i,:).name));
        
        %reverseStr = repmat(sprintf("\b"), 1, length(msg));
        msg = sprintf("\t\tSlices readed: %d/%d\n", i, files_num);
        %fprintf([reverseStr, msg]);
		fprintf(msg);
    end
    fprintf("\n");
end