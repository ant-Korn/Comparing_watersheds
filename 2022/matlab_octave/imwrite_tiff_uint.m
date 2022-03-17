function imwrite_tiff_uint(im, gname)

assert(isnumeric(im));
assert(isinteger(im));
assert(ismatrix(im));
assert(~isempty(im));

t = Tiff(gname,'w');
t.setTag('Photometric',Tiff.Photometric.MinIsBlack);
t.setTag('Compression',Tiff.Compression.LZW);
t.setTag('BitsPerSample',numel(typecast(im(1),'uint8'))*8);
t.setTag('SamplesPerPixel',1);
t.setTag('SampleFormat',Tiff.SampleFormat.UInt);
t.setTag('ImageLength',size(im,1));
t.setTag('ImageWidth',size(im,2));
t.setTag('TileLength',32);
t.setTag('TileWidth',32);
t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
t.write(im);
t.close();

end

