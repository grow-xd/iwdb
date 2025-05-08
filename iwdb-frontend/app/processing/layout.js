import React from 'react';

export const metadata = {
    title: "Processing",
    description: "The users will be redirected here in case any link is being actively processed which is initiated by the user"
}

export default function ProcessingLayout({children}) {
    return (
        <div className="">
            {children}
        </div>
    )
}