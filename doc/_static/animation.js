var timer = null

function set_animation(img_url, timeline, canvas_id)
{
	var canvas = document.getElementById(canvas_id)
	var ctx = canvas.getContext("2d")
	var img = new Image()
	img.onload = function()
	{
        	var i = 0
        	var f = function()
        	{
        		var frame = i++ % timeline.length
        		var delay = timeline[frame].delay
        		var blits = timeline[frame].blit
           		for (j = 0; j < blits.length; ++j)
        		{
        			var blit = blits[j]
        			var sx = blit[0]
        			var sy = blit[1]
        			var w = blit[2]
        			var h = blit[3]
        			var dx = blit[4]
        			var dy = blit[5]
        			ctx.drawImage(img, sx, sy, w, h, dx, dy, w, h)
        		}
        		timer = window.setTimeout(f, delay)
        	}
       	f()
	}
	img.src = img_url
	img.onunload = function()
	{
		if (timer) window.clearTimeout(timer)
	}
}

function set_static(img_url, canvas_id)
{
	var canvas = document.getElementById(canvas_id)
	var ctx = canvas.getContext("2d")
	var img = new Image()
	img.src = img_url
	img.onload = function()
	{
	    	ctx.drawImage(img, 0, 0)
	}
}
